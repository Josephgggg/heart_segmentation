"""Per-patient segmentation metrics (Dice + HD95) and their summary statistics.

Why this exists separately from evaluate.py:

evaluate.py scores slice-by-slice and averages over batches. That is fine as a
cheap training signal, but it is not a number you can report, for two reasons:

  1. A class absent from a slice scores Dice = 1.0 in dice_coeff (eps/eps), and
     33.5% of (slice, class) pairs in this dataset are empty. Slice-wise means
     are therefore inflated. All four structures appear in 100% of *volumes*, so
     scoring whole volumes removes the problem instead of papering over it.
  2. Hausdorff distance is a physical distance and only makes sense in mm on the
     real 3D geometry, not on a stack of independent slices.

So everything here works on whole patient volumes, and the unit of analysis is
the PATIENT. Slices within a patient are highly correlated, so a standard
deviation taken over slices is misleadingly small; take it over patients.
"""

import csv
import logging
import numpy as np
import torch
from collections import defaultdict
from pathlib import Path
from scipy.ndimage import binary_erosion, distance_transform_edt, generate_binary_structure
from torch.utils.data import DataLoader, Subset


def dice_binary(pred: np.ndarray, gt: np.ndarray) -> float:
    """Dice for one binary volume pair. Returns nan if the class is absent from the ground truth."""
    gt_n, pred_n = int(gt.sum()), int(pred.sum())
    if gt_n == 0:
        return float('nan')          # undefined, must not be counted as a perfect score
    return 2.0 * float(np.logical_and(pred, gt).sum()) / (pred_n + gt_n)


def precision_recall(pred: np.ndarray, gt: np.ndarray) -> dict:
    """Precision and recall for one binary volume pair.

    Dice is the harmonic mean of these two, so it cannot distinguish
    over-segmentation from under-segmentation: a model that floods a chamber and
    one that starves it can score the same. Splitting them says which:

        recall    low  -> under-segmenting, the model is missing true voxels
        precision low  -> over-segmenting, the model is claiming voxels it shouldn't

    That is also the direct readout of an asymmetric loss. `tversky_ce` weights
    false negatives above false positives (beta > alpha) specifically to raise
    recall, so without these two you can see that arm's Dice move but not whether
    it moved for the intended reason.

    nan conventions match dice_binary: recall is undefined with no ground truth,
    precision is undefined with no prediction. Note that a total miss (ground
    truth present, prediction empty) gives recall 0.0 and precision nan -- the
    0.0 is real and should count, which is why they are handled separately.
    """
    gt_n, pred_n = int(gt.sum()), int(pred.sum())
    tp = float(np.logical_and(pred, gt).sum())
    return {
        'precision': tp / pred_n if pred_n else float('nan'),
        'recall': tp / gt_n if gt_n else float('nan'),
    }


def _surface(mask: np.ndarray) -> np.ndarray:
    footprint = generate_binary_structure(mask.ndim, 1)
    return mask ^ binary_erosion(mask, structure=footprint, iterations=1)


def _crop_to_union(a: np.ndarray, b: np.ndarray, pad: int = 2):
    """Crop both volumes to the union bounding box (+pad).

    The distance transform is O(volume), and these structures occupy ~1-2% of the
    voxels, so this is a large speedup. It is exact: every surface voxel of both
    masks lies inside the crop, so nearest-surface distances within it are correct.
    The padding guarantees a background border, which keeps the erosion honest for
    structures that would otherwise touch the crop edge.
    """
    union = a | b
    if not union.any():
        return a, b
    slices = []
    for axis, n in enumerate(union.shape):
        idx = np.any(union, axis=tuple(i for i in range(union.ndim) if i != axis)).nonzero()[0]
        slices.append(slice(max(0, idx[0] - pad), min(n, idx[-1] + 1 + pad)))
    slices = tuple(slices)
    return a[slices], b[slices]


def _bidirectional_surface_distances(pred: np.ndarray, gt: np.ndarray, spacing):
    """Surface distances pred->gt and gt->pred, in mm. None if either mask is empty."""
    pred, gt = np.asarray(pred, dtype=bool), np.asarray(gt, dtype=bool)
    if not pred.any() or not gt.any():
        return None

    pred_c, gt_c = _crop_to_union(pred, gt)
    pred_surf, gt_surf = _surface(pred_c), _surface(gt_c)
    if not pred_surf.any() or not gt_surf.any():
        return None

    d_pred_to_gt = distance_transform_edt(~gt_surf, sampling=spacing)[pred_surf]
    d_gt_to_pred = distance_transform_edt(~pred_surf, sampling=spacing)[gt_surf]
    return d_pred_to_gt, d_gt_to_pred


def surface_metrics(pred: np.ndarray, gt: np.ndarray, spacing) -> dict:
    """HD95 and ASSD in mm, sharing one pair of distance transforms.

    Both follow the medpy convention, so they are comparable with the numbers
    reported by the ACDC / M&Ms / BraTS challenge literature:
      HD95 -- 95th percentile of the POOLED bidirectional distances
              (not the max of the two one-directional percentiles).
      ASSD -- mean of the two directional means
              (not the mean of the pooled distances; these differ when the two
              surfaces have different voxel counts).

    Both are nan when either mask is empty -- the distance is genuinely undefined
    there. An empty prediction against a non-empty ground truth is a total miss,
    not a distance of 0, so callers must count those separately rather than
    letting them silently drop out of a mean.
    """
    d = _bidirectional_surface_distances(pred, gt, spacing)
    if d is None:
        return {'hd95_mm': float('nan'), 'assd_mm': float('nan')}
    d_pred_to_gt, d_gt_to_pred = d
    return {
        'hd95_mm': float(np.percentile(np.hstack((d_pred_to_gt, d_gt_to_pred)), 95)),
        'assd_mm': float(np.mean([d_pred_to_gt.mean(), d_gt_to_pred.mean()])),
    }


def hd95(pred: np.ndarray, gt: np.ndarray, spacing) -> float:
    """95th-percentile symmetric Hausdorff distance in mm. See surface_metrics()."""
    return surface_metrics(pred, gt, spacing)['hd95_mm']


def assd(pred: np.ndarray, gt: np.ndarray, spacing) -> float:
    """Average symmetric surface distance in mm. See surface_metrics()."""
    return surface_metrics(pred, gt, spacing)['assd_mm']


@torch.inference_mode()
@torch.inference_mode()
def predict_volume(net, dataset, patient_indices, device, amp=False, batch_size=8,
                   num_workers=2, return_entropy=False):
    """Run the model over one patient's slices, in slice order, and stack to 3D.

    return_entropy=True also returns the per-pixel Shannon entropy of the softmax
    distribution (nats; 0 = fully confident, up to log(n_classes) = uniform over
    every class) -- one extra softmax + reduction on logits already computed for
    the argmax, not an extra forward pass. Computed in fp32 regardless of `amp`,
    same reasoning as the loss: log() of small probabilities is the numerically
    fragile part, and autocast's fp16 doesn't reliably keep that accurate.
    """
    ordered = sorted(patient_indices, key=lambda i: dataset.index[i][1])
    loader = DataLoader(Subset(dataset, ordered), batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True)
    preds, gts, entropies = [], [], []
    was_training = net.training
    net.eval()
    for batch in loader:
        images = batch['image'].to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
        with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
            logits = net(images)
        preds.append(logits.argmax(dim=1).cpu().numpy().astype(np.uint8))
        gts.append(batch['mask'].numpy().astype(np.uint8))
        if return_entropy:
            probs = torch.softmax(logits.float(), dim=1)
            entropy = -(probs * torch.log(probs.clamp_min(1e-12))).sum(dim=1)
            entropies.append(entropy.cpu().numpy().astype(np.float32))
    if was_training:
        net.train()
    if return_entropy:
        return np.concatenate(preds), np.concatenate(gts), np.concatenate(entropies)
    return np.concatenate(preds), np.concatenate(gts)


def evaluate_per_patient(net, dataset, indices, device, n_classes, amp=False,
                         class_names=None, batch_size=8, compute_hd95=True, quiet=False):
    """Score every patient covered by `indices`, one row per (patient, class).

    Returns a list of dicts: patient_id, cls, class_name, dice, hd95_mm,
    gt_voxels, pred_voxels, missed (prediction empty while ground truth is not).

    `compute_hd95=False` skips the surface distances (and the DICOM header read
    they need), leaving just Dice -- that is the cheap path used for per-epoch
    checkpoint selection. `quiet` suppresses the per-patient progress line, which
    is noise when this runs every epoch.
    """
    by_patient = defaultdict(list)
    for i in indices:
        by_patient[dataset.index[i][0]].append(i)

    rows = []
    for n, (patient_id, idxs) in enumerate(sorted(by_patient.items()), start=1):
        pred_vol, gt_vol = predict_volume(net, dataset, idxs, device, amp=amp, batch_size=batch_size)
        spacing = dataset.spacing_for(patient_id) if compute_hd95 else None
        if not quiet:
            logging.info(f'  [{n}/{len(by_patient)}] {patient_id}: {pred_vol.shape[0]} slices, '
                         f'spacing {tuple(round(s, 4) for s in spacing)} mm')

        for cls in range(1, n_classes):       # class 0 is background
            pred_c, gt_c = pred_vol == cls, gt_vol == cls
            surf = (surface_metrics(pred_c, gt_c, spacing) if compute_hd95
                    else {'hd95_mm': float('nan'), 'assd_mm': float('nan')})
            pr = precision_recall(pred_c, gt_c)
            rows.append({
                'patient_id': patient_id,
                'cls': cls,
                'class_name': (class_names or {}).get(cls, f'class_{cls}'),
                'dice': dice_binary(pred_c, gt_c),
                'precision': pr['precision'],
                'recall': pr['recall'],
                'hd95_mm': surf['hd95_mm'],
                'assd_mm': surf['assd_mm'],
                'gt_voxels': int(gt_c.sum()),
                'pred_voxels': int(pred_c.sum()),
                # for volume agreement; needs spacing, so nan on the cheap path
                'voxel_ml': float(np.prod(spacing)) / 1000.0 if spacing is not None else float('nan'),
                'missed': bool(gt_c.any() and not pred_c.any()),
            })
    return rows


def macro_dice(net, dataset, indices, device, n_classes, amp=False, batch_size=8):
    """Per-patient, per-class Dice on whole volumes, averaged to one number.

    This is the checkpoint-selection signal. `evaluate.py`'s slice-wise Dice is
    unusable for that in a loss ablation: it scores a class absent from a slice
    as 1.0, so it rewards a model for predicting nothing on empty slices -- and
    how a loss treats empty classes is exactly what the arms differ in. Selecting
    on it would fold that bias into the comparison.

    Each patient is reduced to a mean over the foreground classes first, then
    averaged over patients, so a patient counts once regardless of slice count.
    """
    rows = evaluate_per_patient(net, dataset, indices, device, n_classes, amp=amp,
                                batch_size=batch_size, compute_hd95=False, quiet=True)
    per_patient = defaultdict(list)
    for r in rows:
        per_patient[r['patient_id']].append(r['dice'])
    means = [np.nanmean(v) for v in per_patient.values() if not np.all(np.isnan(v))]
    return float(np.mean(means)) if means else float('nan')


DEFAULT_METRICS = ('dice', 'precision', 'recall', 'hd95_mm', 'assd_mm')


def summarise(rows, metrics=DEFAULT_METRICS):
    """Mean / SD / median / IQR across PATIENTS, per class, plus an all-class row.

    n is the number of patients actually contributing (nan values are excluded),
    so you can see when a statistic rests on very few patients. `n_missed` counts
    patients where the structure exists but the model predicted nothing for it --
    those have no defined HD95 and would otherwise vanish from the mean, making
    the model look better the more badly it fails.

    Metrics absent from `rows` are dropped rather than raising, so CSVs written
    before a metric existed (the pre-ASSD, pre-precision/recall runs) still load.
    """
    out = []
    metrics = tuple(m for m in metrics if rows and m in rows[0])
    by_class = defaultdict(list)
    for r in rows:
        by_class[(r['cls'], r['class_name'])].append(r)

    def stats(vals):
        vals = np.asarray([v for v in vals if not np.isnan(v)], dtype=float)
        if vals.size == 0:
            return dict(n=0, mean=float('nan'), sd=float('nan'),
                        median=float('nan'), q1=float('nan'), q3=float('nan'))
        return dict(n=int(vals.size), mean=float(vals.mean()),
                    sd=float(vals.std(ddof=1)) if vals.size > 1 else float('nan'),
                    median=float(np.median(vals)),
                    q1=float(np.percentile(vals, 25)), q3=float(np.percentile(vals, 75)))

    for (cls, name), group in sorted(by_class.items()):
        row = {'cls': cls, 'class_name': name,
               'n_patients': len(group),
               'n_missed': sum(r['missed'] for r in group)}
        for m in metrics:
            for k, v in stats([r[m] for r in group]).items():
                row[f'{m}_{k}'] = v
        out.append(row)

    # macro average over classes, computed per patient first so each patient counts once
    per_patient = defaultdict(dict)
    for r in rows:
        for m in metrics:
            per_patient[r['patient_id']].setdefault(m, []).append(r[m])
    row = {'cls': -1, 'class_name': 'all_classes_macro',
           'n_patients': len(per_patient),
           'n_missed': sum(r['missed'] for r in rows)}
    for m in metrics:
        means = [np.nanmean(v[m]) if not np.all(np.isnan(v[m])) else float('nan')
                 for v in per_patient.values()]
        for k, v in stats(means).items():
            row[f'{m}_{k}'] = v
    out.append(row)
    return out


def format_summary(summary, metrics=DEFAULT_METRICS):
    """Human-readable 'mean +/- SD' table, ready to paste into notes or a paper."""
    metrics = tuple(m for m in metrics if summary and f'{m}_n' in summary[0])

    def cell(m, r):
        if not r[f'{m}_n']:
            return f'{"n/a":>21}'
        dp = 4 if m in ('dice', 'precision', 'recall') else 2
        return f'{r[f"{m}_mean"]:>10.{dp}f} +/- {r[f"{m}_sd"]:<8.{dp}f}'

    lines = [f'{"class":<20}{"n":>4}{"missed":>8}' + ''.join(f'{m:>21}' for m in metrics)]
    for r in summary:
        lines.append(f'{r["class_name"]:<20}{r["n_patients"]:>4}{r["n_missed"]:>8}'
                     + ''.join(cell(m, r) for m in metrics))
    return '\n'.join(lines)


def load_per_patient(path):
    """Read a *_metrics_per_patient.csv back, with numeric columns parsed."""
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            for k in ('dice', 'precision', 'recall', 'hd95_mm', 'assd_mm', 'voxel_ml'):
                if k in r:
                    r[k] = float(r[k]) if r[k] not in ('', 'nan') else float('nan')
            for k in ('cls', 'gt_voxels', 'pred_voxels'):
                if k in r and r[k] != '':
                    r[k] = int(r[k])
            if 'missed' in r:
                r['missed'] = str(r['missed']).lower() == 'true'
            rows.append(r)
    return rows


def _holm(pvals):
    """Holm-Bonferroni step-down adjusted p-values, returned in the input order."""
    m = len(pvals)
    adj, running = [0.0] * m, 0.0
    for rank, i in enumerate(sorted(range(m), key=lambda j: pvals[j])):
        running = max(running, (m - rank) * pvals[i])
        adj[i] = min(1.0, running)
    return adj


def compare_runs(path_a, path_b, label_a='A', label_b='B',
                 # deliberately NOT DEFAULT_METRICS: precision and recall are
                 # diagnostics for reading a single run, not endpoints. Adding
                 # them would grow the Holm family from 12 tests to 20 and weaken
                 # the correction on the metrics you actually want to compare.
                 # Pass them explicitly if you specifically want to test them.
                 metrics=('dice', 'hd95_mm', 'assd_mm')):
    """Paired Wilcoxon signed-rank comparison of two runs scored on the SAME patients.

    Pairs rows on (patient_id, class_name). Paired is the right test here: both
    models score the same patients, and pairing removes between-patient variance,
    which dominates everything else at this sample size.

    IMPORTANT sample-size limit. The two-sided exact Wilcoxon test on n pairs
    cannot return a p below 2 / 2**n. At n = 7 patients that floor is 0.0156, so:
      - a single pre-specified test (use the all_classes_macro row) CAN reach
        p < 0.05, but only if every patient moves the same way;
      - the 12 per-chamber x per-metric tests CANNOT, because Holm multiplies the
        smallest p by 12 -> 0.19. Treat those as descriptive, not confirmatory.
    Holm correction is therefore applied across the per-chamber tests only, and
    the macro row is left uncorrected as the single primary endpoint.

    Returns a list of dicts, one per (class, metric), macro row last.
    """
    from scipy.stats import wilcoxon

    a = {(r['patient_id'], r['class_name']): r for r in load_per_patient(path_a)}
    b = {(r['patient_id'], r['class_name']): r for r in load_per_patient(path_b)}
    shared = sorted(set(a) & set(b))
    if not shared:
        raise ValueError('No shared (patient, class) pairs between the two runs. '
                         'Were they scored on the same split? Check split_seed.')
    if set(a) - set(b) or set(b) - set(a):
        logging.warning(f'{len(set(a) - set(b))} pairs only in {label_a}, '
                        f'{len(set(b) - set(a))} only in {label_b}; '
                        f'comparing the {len(shared)} shared pairs only')

    def test(va, vb):
        va, vb = np.asarray(va, float), np.asarray(vb, float)
        ok = ~(np.isnan(va) | np.isnan(vb))
        va, vb = va[ok], vb[ok]
        p = float('nan')
        if va.size and np.any(vb - va):
            try:
                p = float(wilcoxon(va, vb).pvalue)
            except ValueError:
                p = float('nan')
        return va, vb, p

    out = []
    for cls in sorted({c for _, c in shared}):
        keys = [k for k in shared if k[1] == cls]
        for m in metrics:
            va, vb, p = test([a[k][m] for k in keys], [b[k][m] for k in keys])
            out.append({'class_name': cls, 'metric': m, 'n_pairs': int(va.size),
                        f'median_{label_a}': float(np.median(va)) if va.size else float('nan'),
                        f'median_{label_b}': float(np.median(vb)) if vb.size else float('nan'),
                        'median_diff': float(np.median(vb - va)) if va.size else float('nan'),
                        'p_value': p, 'primary': False})

    # Holm across the per-chamber family only
    idx = [i for i, r in enumerate(out) if not np.isnan(r['p_value'])]
    for r in out:
        r['p_holm'] = float('nan')
    for i, v in zip(idx, _holm([out[i]['p_value'] for i in idx])):
        out[i]['p_holm'] = v

    # macro: average each patient over chambers first, then one test per metric
    patients = sorted({p for p, _ in shared})
    for m in metrics:
        ma = [np.nanmean([a[k][m] for k in shared if k[0] == p]) for p in patients]
        mb = [np.nanmean([b[k][m] for k in shared if k[0] == p]) for p in patients]
        va, vb, p = test(ma, mb)
        out.append({'class_name': 'all_classes_macro', 'metric': m, 'n_pairs': int(va.size),
                    f'median_{label_a}': float(np.median(va)) if va.size else float('nan'),
                    f'median_{label_b}': float(np.median(vb)) if vb.size else float('nan'),
                    'median_diff': float(np.median(vb - va)) if va.size else float('nan'),
                    'p_value': p, 'p_holm': float('nan'), 'primary': True})
    return out


def format_comparison(comparison, label_a='A', label_b='B'):
    """Readable paired-comparison table. '*' marks the pre-specified primary endpoint."""
    ka, kb = f'median_{label_a}', f'median_{label_b}'
    # widen the two median columns to fit the labels -- loss-arm names like
    # "dice_ce_boundary" overflow a fixed 12 and run into the next header
    w = max(12, len(label_a) + 2, len(label_b) + 2)
    lines = [f'{"class":<20}{"metric":<10}{"n":>3}{label_a:>{w}}{label_b:>{w}}'
             f'{"diff":>10}{"p":>10}{"p_holm":>10}',
             f'{"-" * (63 + 2 * w)}']
    for r in comparison:
        dp = 4 if r['metric'] == 'dice' else 2
        ph = f'{r["p_holm"]:.4f}' if not np.isnan(r['p_holm']) else ('primary' if r['primary'] else '-')
        pv = f'{r["p_value"]:.4f}' if not np.isnan(r['p_value']) else 'n/a'
        star = '*' if r['primary'] else ' '
        lines.append(f'{star + r["class_name"]:<20}{r["metric"]:<10}{r["n_pairs"]:>3}'
                     f'{r[ka]:>{w}.{dp}f}{r[kb]:>{w}.{dp}f}{r["median_diff"]:>+10.{dp}f}'
                     f'{pv:>10}{ph:>10}')
    return '\n'.join(lines)


def write_csv(rows, path):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    logging.info(f'Wrote {len(rows)} rows to {path}')


def report(net, dataset, indices, device, n_classes, out_dir=None, split_name='test',
           amp=False, class_names=None, batch_size=8, compute_hd95=True):
    """Score a split per patient, write both CSVs, and return (rows, summary)."""
    logging.info(f'Scoring {split_name} split per patient (Dice + {"HD95" if compute_hd95 else "no HD95"})...')
    rows = evaluate_per_patient(net, dataset, indices, device, n_classes, amp=amp,
                                class_names=class_names, batch_size=batch_size,
                                compute_hd95=compute_hd95)
    summary = summarise(rows)
    logging.info(f'\n{split_name} results (mean +/- SD across patients):\n{format_summary(summary)}')
    if out_dir is not None:
        write_csv(rows, out_dir / f'{split_name}_metrics_per_patient.csv')
        write_csv(summary, out_dir / f'{split_name}_metrics_summary.csv')
    return rows, summary


def _plot_overlay(img, gt, pred, n_classes, class_names, title, path):
    """Save one PNG: input image with GT and predicted masks overlaid side by side."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    colors = ['none', 'tab:red', 'tab:green', 'tab:blue', 'tab:orange', 'tab:purple']
    cmap = ListedColormap(colors[:n_classes])
    class_names = class_names or {}

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))
    for ax, mask, label in zip(axes, (gt, pred), ('ground truth', 'prediction')):
        ax.imshow(img, cmap='gray')
        ax.imshow(np.ma.masked_equal(mask, 0), cmap=cmap, vmin=0, vmax=n_classes - 1, alpha=0.5)
        ax.set_title(label)
        ax.axis('off')

    handles = [Patch(color=colors[c], label=class_names.get(c, f'class_{c}')) for c in range(1, n_classes)]
    fig.legend(handles=handles, loc='lower center', ncol=len(handles), frameon=False)
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_best_worst_slices(net, dataset, indices, device, n_classes, out_dir,
                            class_names=None, amp=False, batch_size=8, split_name='val'):
    """Find and plot the best- and worst-scoring slices in a split, for one model.

    Per-slice score is the mean Dice over foreground classes present in that
    slice's ground truth. Slices with no foreground class are skipped entirely --
    otherwise an empty slice trivially scores 1.0 (`dice_binary`'s eps/eps case,
    same inflation `evaluate.py` has at the volume level) and would dominate
    "best" with a meaningless result.

    Saves `<split_name>_best_slice.png` / `<split_name>_worst_slice.png` under
    `out_dir` (the run's checkpoint directory) and returns their paths, or None
    if the split has no slice with any foreground.
    """
    by_patient = defaultdict(list)
    for i in indices:
        by_patient[dataset.index[i][0]].append(i)

    best = worst = None   # (score, patient_id, slice_idx, global_idx, pred_slice, gt_slice)
    for patient_id, idxs in sorted(by_patient.items()):
        ordered = sorted(idxs, key=lambda i: dataset.index[i][1])
        pred_vol, gt_vol = predict_volume(net, dataset, idxs, device, amp=amp, batch_size=batch_size)
        for pos, global_idx in enumerate(ordered):
            gt_slice, pred_slice = gt_vol[pos], pred_vol[pos]
            class_scores = [dice_binary(pred_slice == c, gt_slice == c) for c in range(1, n_classes)]
            class_scores = [s for s in class_scores if not np.isnan(s)]
            if not class_scores:
                continue   # no foreground in this slice's ground truth -- trivial, skip
            score = float(np.mean(class_scores))
            entry = (score, patient_id, dataset.index[global_idx][1], global_idx, pred_slice, gt_slice)
            if best is None or score > best[0]:
                best = entry
            if worst is None or score < worst[0]:
                worst = entry

    if best is None:
        logging.warning(f'{split_name}: no slice with foreground found, skipping best/worst slice plots')
        return None

    out_dir = Path(out_dir)
    paths = {}
    for tag, entry in (('best', best), ('worst', worst)):
        score, patient_id, slice_idx, global_idx, pred_slice, gt_slice = entry
        img = dataset[global_idx]['image'].numpy()[0]
        path = out_dir / f'{split_name}_{tag}_slice.png'
        _plot_overlay(img, gt_slice, pred_slice, n_classes, class_names,
                      title=f'{tag} {split_name} slice — {patient_id} #{slice_idx} (Dice={score:.3f})',
                      path=path)
        logging.info(f'{split_name} {tag} slice: {patient_id} #{slice_idx}, Dice={score:.3f} -> {path}')
        paths[tag] = path
    return paths
