"""Export this project's cardiac MRI volumes into nnU-Net v2 raw dataset format.

nnU-Net is a framework, not a model: it owns preprocessing, resampling, patch-size
and architecture planning, training and inference. It therefore cannot be an
`--arch` in train.py -- it runs as a separate pipeline, in its own venv, on data
exported into its layout, and is bridged back into this repo at the metrics layer
(see score_nnunet_predictions.py).

Source of truth is `data/preprocessed_cache/*_both_{img,mask}.npy`, i.e. exactly
what `VolumeMRIDataset` built and what every `both_*` run trained on. Reusing the
cache rather than re-reading DICOM/NIfTI means the exported data is provably the
same data, and it already has:
  * the half-volume water/fat split applied (one .dcm holds water then fat),
  * the mask reorientation (transpose(2,0,1) -> x-flip -> rot90),
  * the phase label handling ('both' keeps labels 0-5, EAT stays class 5).

Four things this script is careful about:

  * RAW intensities. nnU-Net does its own per-image z-score normalisation and
    that is part of the method, so `BasicDataset.preprocess` is deliberately NOT
    applied -- normalising here would double-normalise and strip out the thing
    being tested. (Intensity scale is not even consistent across this cohort:
    some volumes are 0-255, others run past 1700 with fractional values, which
    is exactly what per-image normalisation is for. Stored float32; casting to
    uint8 would be lossy for those patients.)
  * NATIVE 432x432, not `--scale 0.5`. nnU-Net picks its own target spacing;
    handing it a pre-downsampled grid removes one of the things being tested.
  * CORRECT affines. The source NIfTI headers are junk -- nibabel reports zooms
    of ~(696759, 696759, 1250000) -- which is why this repo reads spacing from
    the DICOM instead. Spacing is taken per patient from
    `VolumeMRIDataset.spacing_for()` at scale=1.0 and is NOT uniform across the
    cohort, so it is never hardcoded.
  * TEST PATIENTS ARE NOT EXPORTED AS TRAINING DATA. The held-out test patients
    go to imagesTs, which nnU-Net never preprocesses for training. splits_final
    alone would keep them out, but a directory boundary makes leakage
    structurally impossible rather than merely configured.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train import BOTH_CLASS_NAMES                      # noqa: E402
from utils.data_loading import VolumeMRIDataset         # noqa: E402

PHASE = 'both'
N_CLASSES = 6                    # 0 background + LV/RV/LA/RA/EAT


def load_patient_lists(run_config_path):
    """Pool (trainable) and test patient ids, taken verbatim from a control run.

    Reading the lists off a finished run rather than re-deriving the shuffle is
    deliberate: `split_patients_kfold` is seeded and reproducible, but the run
    configs are the actual record of what the control arm trained on, so there
    is no way for the two to silently drift apart.
    """
    cfg = json.loads(Path(run_config_path).read_text())
    if cfg.get('split_seed') != 0:
        raise SystemExit(f"expected split_seed 0, got {cfg.get('split_seed')}")
    pool = sorted(set(cfg['train_patients']) | set(cfg['val_patients']))
    test = sorted(cfg['test_patients'])
    overlap = set(pool) & set(test)
    if overlap:
        raise SystemExit(f'pool and test overlap: {sorted(overlap)}')
    return pool, test, cfg


def affine_for(spacing):
    """(Z, Y, X) array spacing -> a diagonal nibabel affine.

    `spacing_for` returns (slice, row, col) and the volumes are (Z, Y, X), so
    axis i takes the slice spacing and axes j/k the in-plane ones. Written
    diagonally on purpose: nnU-Net only needs image and label geometry to agree
    and the voxel sizes to be right, and a diagonal affine makes the round-trip
    check in verify_export() unambiguous.
    """
    sz, sy, sx = spacing
    return np.diag([float(sz), float(sy), float(sx), 1.0])


def export(cache_dir, out_dir, ds, patients, images_sub, labels_sub, write_labels=True):
    img_dir, lbl_dir = out_dir / images_sub, out_dir / labels_sub
    img_dir.mkdir(parents=True, exist_ok=True)
    if write_labels:
        lbl_dir.mkdir(parents=True, exist_ok=True)

    for n, pid in enumerate(patients, 1):
        img = np.load(cache_dir / f'{pid}_{PHASE}_img.npy')     # (Z, 2, 432, 432)
        msk = np.load(cache_dir / f'{pid}_{PHASE}_mask.npy')    # (Z, 432, 432)
        if img.ndim != 4 or img.shape[1] != 2:
            raise SystemExit(f'{pid}: expected (Z, 2, H, W) image cache, got {img.shape}')
        if img.shape[0] != msk.shape[0] or img.shape[2:] != msk.shape[1:]:
            raise SystemExit(f'{pid}: image {img.shape} and mask {msk.shape} disagree')

        aff = affine_for(ds.spacing_for(pid))
        for ch in (0, 1):                                       # 0000 water, 0001 fat
            vol = np.ascontiguousarray(img[:, ch], dtype=np.float32)
            nib.save(nib.Nifti1Image(vol, aff), img_dir / f'{pid}_{ch:04d}.nii.gz')

        if write_labels:
            labels = np.unique(msk)
            if not np.array_equal(labels, np.rint(labels)) or labels.max() >= N_CLASSES:
                raise SystemExit(f'{pid}: unexpected label set {labels}')
            lab = np.ascontiguousarray(msk, dtype=np.uint8)
            nib.save(nib.Nifti1Image(lab, aff), lbl_dir / f'{pid}.nii.gz')

        print(f'  [{n:>2}/{len(patients)}] {pid:22s} {tuple(img.shape)} '
              f'spacing={tuple(round(float(s), 4) for s in ds.spacing_for(pid))}', flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', type=Path,
                    default=Path('/hpc/jgeo610/nnunet/raw/Dataset501_CardiacBoth'))
    ap.add_argument('--cache-dir', type=Path, default=Path('data/preprocessed_cache'))
    ap.add_argument('--imgs', type=Path, default=Path('data/imgs'))
    ap.add_argument('--masks', type=Path, default=Path('data/masks'))
    ap.add_argument('--control-run', type=Path,
                    default=Path('checkpoints/both_data_aug_dice_ce_kfold0/run_config.json'),
                    help='run_config.json the patient lists are taken from')
    ap.add_argument('--limit', type=int, default=0,
                    help='export only the first N pool patients (smoke test)')
    args = ap.parse_args()

    pool, test, cfg = load_patient_lists(args.control_run)
    print(f'control run : {args.control_run}')
    print(f'pool        : {len(pool)} patients (imagesTr)')
    print(f'test        : {len(test)} patients (imagesTs, never preprocessed for training)')

    # scale=1.0 so spacing_for() returns native in-plane spacing rather than the
    # 0.5-corrected value the training runs used.
    ds = VolumeMRIDataset(args.imgs, args.masks, scale=1.0, phase=PHASE)

    if args.limit:
        pool, test = pool[:args.limit], test[:1]
        print(f'--limit {args.limit}: exporting {len(pool)} pool + {len(test)} test')

    print('\nimagesTr / labelsTr:')
    export(args.cache_dir, args.out, ds, pool, 'imagesTr', 'labelsTr', write_labels=True)
    print('\nimagesTs (held out, no labels exported):')
    export(args.cache_dir, args.out, ds, test, 'imagesTs', 'labelsTs', write_labels=False)

    dataset_json = {
        'channel_names': {'0': 'water', '1': 'fat'},
        'labels': {'background': 0, **{v: k for k, v in BOTH_CLASS_NAMES.items()}},
        'numTraining': len(pool),
        'file_ending': '.nii.gz',
        'description': ('Cardiac MRI water/fat, exported from heart_segmentation. '
                        f"Patient split taken from {args.control_run}, split_seed "
                        f"{cfg.get('split_seed')}."),
    }
    (args.out / 'dataset.json').write_text(json.dumps(dataset_json, indent=2))
    print(f"\nwrote {args.out / 'dataset.json'}")
    print(json.dumps(dataset_json['labels'], indent=2))


if __name__ == '__main__':
    main()
