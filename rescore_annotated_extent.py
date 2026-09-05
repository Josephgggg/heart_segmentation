"""Re-score finished runs under three slice-inclusion policies (sensitivity analysis).

WHY
---
The NIfTI masks in this dataset are truncated in z: labelling stops at a slice
where the anatomy plainly continues. 48 of 50 volumes have unlabelled slices at
the top, and on average 16.4% of the slices in a volume fall outside the
annotated block. Ground truth there reads "background everywhere", so an
anatomically correct prediction on those slices is counted as a false positive.

That distorts the numbers in two different ways. It deflates every absolute
score, and -- worse for an ablation -- it rewards whichever arm happens to be
more conservative near the ends of the volume, because guessing where the
annotator stopped is worth real Dice. So part of any arm ranking is a ranking of
protocol-mimicry rather than of segmentation quality.

WHAT THIS IS
------------
A sensitivity analysis, not a correction. The primary endpoint (whole-volume
scoring) is left exactly as it is; this recomputes the same metrics under two
alternative, defensible policies and lets you check whether the conclusions
move. If they do not, the conclusions are robust to the assumption. If they do,
that is itself the finding.

  full       every slice -- the primary, unchanged. Recomputed here purely as an
             integrity check that this script scores identically to train.py.
  annotated  only slices carrying any annotation at all.
  eroded     annotated, shrunk by MARGIN slices at each end. The outermost
             labelled slice is often only partially drawn (the annotator
             tapering off), so this tests whether `annotated` is itself
             sensitive to exactly where the boundary falls.

This re-scores existing best.pth checkpoints. It trains nothing, changes no
configuration, and touches VALIDATION patients only -- the held-out test set is
never read.
"""
import argparse, json, logging, sys
from pathlib import Path

import numpy as np
import torch

from train import build_model, n_channels_for, BOTH_CLASS_NAMES, CLASS_NAMES, dir_img, dir_mask
from utils import metrics
from utils.data_loading import VolumeMRIDataset

MARGIN = 3
VARIANTS = {                       # name -> (restrict_to_annotated, margin)
    'full':      (False, 0),
    'annotated': (True,  0),
    'eroded':    (True,  MARGIN),
}
DEFAULT_ARMS = ['both_baseline_dice_ce', 'both_data_aug_dice_ce',
                'both_ctx_cs1', 'both_ctx_cs2', 'both_ctx_cs3', 'both_attn']

_datasets = {}


def get_dataset(cfg):
    key = (cfg['phase'], cfg.get('context_slices', 0), cfg['img_scale'])
    if key not in _datasets:
        _datasets[key] = VolumeMRIDataset(dir_img, dir_mask, scale=cfg['img_scale'],
                                          phase=cfg['phase'],
                                          context_slices=cfg.get('context_slices', 0))
    return _datasets[key]


def score_run(run_dir, device, force=False):
    cfg = json.loads((run_dir / 'run_config.json').read_text())
    out = {v: run_dir / f'val_metrics_per_patient_{v}.csv' for v in VARIANTS}
    if not force and all(p.exists() for p in out.values()):
        logging.info(f'{run_dir.name}: already scored, skipping')
        return

    ds = get_dataset(cfg)
    net = build_model(cfg.get('arch') or 'unet',
                      n_channels_for(cfg['phase'], cfg.get('context_slices', 0)),
                      cfg['n_classes'], bilinear=cfg.get('bilinear', False))
    sd = torch.load(run_dir / 'best.pth', map_location='cpu')
    sd.pop('mask_values', None)          # train.py injects this; load_state_dict would choke
    net.load_state_dict(sd)
    net = net.to(device).eval()

    class_names = BOTH_CLASS_NAMES if cfg['phase'] == 'both' else CLASS_NAMES
    val = set(cfg['val_patients'])
    assert not (val & set(cfg['test_patients'])), f'{run_dir.name}: val/test overlap'
    by_patient = {}
    for i, (pid, _) in enumerate(ds.index):
        if pid in val:
            by_patient.setdefault(pid, []).append(i)
    assert set(by_patient) == val, f'{run_dir.name}: missing val patients {val - set(by_patient)}'

    rows = {v: [] for v in VARIANTS}
    for n, (pid, idxs) in enumerate(sorted(by_patient.items()), start=1):
        # ONE inference pass, scored three ways -- that is why rows_for_volume exists
        pred_vol, gt_vol = metrics.predict_volume(net, ds, idxs, device, amp=False, batch_size=8)
        spacing = ds.spacing_for(pid)
        for v, (restrict, margin) in VARIANTS.items():
            keep = (metrics.annotated_slices(gt_vol, margin=margin) if restrict
                    else np.ones(gt_vol.shape[0], bool))
            rows[v] += metrics.rows_for_volume(
                pred_vol[keep], gt_vol[keep], pid, cfg['n_classes'], spacing=spacing,
                class_names=class_names, compute_hd95=True,
                extra={'n_slices': int(gt_vol.shape[0]), 'n_scored': int(keep.sum())})
        logging.info(f'  [{n}/{len(by_patient)}] {pid}: {gt_vol.shape[0]} slices -> '
                     + ', '.join(f'{v} {rows[v][-1]["n_scored"]}' for v in VARIANTS))

    for v in VARIANTS:
        metrics.write_csv(rows[v], out[v])
        metrics.write_csv(metrics.summarise(rows[v]), run_dir / f'val_metrics_summary_{v}.csv')

    # integrity: `full` must reproduce the CSV train.py already wrote
    original = run_dir / 'val_metrics_per_patient.csv'
    if original.exists():
        was = {(r['patient_id'], r['cls']): r['dice'] for r in metrics.load_per_patient(original)}
        now = {(r['patient_id'], r['cls']): r['dice'] for r in rows['full']}
        shared = set(was) & set(now)
        worst = max((abs(was[k] - now[k]), k) for k in shared) if shared else (0.0, None)
        logging.info(f'{run_dir.name}: full vs recorded CSV, max |Delta dice| = {worst[0]:.2e} '
                     f'over {len(shared)} (patient, class) pairs')
        assert worst[0] < 5e-3, f'{run_dir.name}: re-scoring does not reproduce the recorded Dice'


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('arms', nargs='*', default=DEFAULT_ARMS,
                    help='run-name prefixes; each expands to <prefix>_kfold*')
    ap.add_argument('--force', action='store_true', help='re-score even if CSVs exist')
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'device: {device}')

    run_dirs = [d for arm in args.arms
                for d in sorted(Path('checkpoints').glob(f'{arm}_kfold*'))
                if (d / 'run_config.json').exists() and (d / 'best.pth').exists()]
    if not run_dirs:
        sys.exit(f'no runs found for {args.arms}')
    logging.info(f'{len(run_dirs)} run directories to score')
    for d in run_dirs:
        logging.info(f'=== {d.name} ===')
        score_run(d, device, force=args.force)
    logging.info('done')


if __name__ == '__main__':
    main()
