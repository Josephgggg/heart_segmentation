"""Write nnU-Net's splits_final.json from this project's finished control runs.

nnU-Net generates its own 5-fold split unless `splits_final.json` already exists
in the preprocessed folder, in which case it uses that. Injecting the folds is
what makes the whole comparison possible: `metrics.compare_runs` pairs on
(patient_id, class_name), so the nnU-Net arm and the control arm have to be
evaluated on exactly the same patients in exactly the same folds, or there is
nothing to pair.

The fold membership is taken verbatim from
`checkpoints/both_data_aug_dice_ce_kfold<F>/run_config.json` rather than being
re-derived from `split_patients_kfold`. Both would agree today -- the shuffle is
seeded -- but the run configs are the record of what the control arm actually
trained on, so reading them removes any way for the two to drift apart.

Run this AFTER nnUNetv2_plan_and_preprocess (which creates the target folder)
and BEFORE nnUNetv2_train.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

DEFAULT_RUNS = 'checkpoints/both_data_aug_dice_ce_kfold{fold}/run_config.json'


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--preprocessed', type=Path,
                    default=Path('/hpc/jgeo610/nnunet/preprocessed/Dataset501_CardiacBoth'))
    ap.add_argument('--runs', default=DEFAULT_RUNS,
                    help="run_config.json path template containing '{fold}'")
    ap.add_argument('--k-folds', type=int, default=5)
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()

    splits, all_val, test_ref, pool_ref = [], set(), None, None
    for fold in range(args.k_folds):
        cfg = json.loads(Path(args.runs.format(fold=fold)).read_text())
        if cfg.get('split_seed') != 0 or cfg.get('fold') != fold:
            raise SystemExit(f'fold {fold}: unexpected split_seed/fold in {cfg.get("run_name")}')
        train, val = sorted(cfg['train_patients']), sorted(cfg['val_patients'])
        pool = set(train) | set(val)

        if pool_ref is None:
            pool_ref, test_ref = pool, set(cfg['test_patients'])
        elif pool != pool_ref:
            raise SystemExit(f'fold {fold}: trainable pool differs from fold 0')
        if set(train) & set(val):
            raise SystemExit(f'fold {fold}: train and val overlap')
        if pool & test_ref:
            raise SystemExit(f'fold {fold}: test patients leaked into the pool')

        splits.append({'train': train, 'val': val})
        if all_val & set(val):
            raise SystemExit(f'fold {fold}: val overlaps an earlier fold')
        all_val |= set(val)
        print(f'  fold {fold}: train={len(train):3d}  val={len(val):3d}')

    if all_val != pool_ref:
        missing, extra = pool_ref - all_val, all_val - pool_ref
        raise SystemExit(f'val folds do not partition the pool (missing={missing}, extra={extra})')

    print(f'\npool partitioned by val folds: {len(all_val)} patients')
    print(f'held-out test (never in any split): {len(test_ref)} patients')

    out = args.preprocessed / 'splits_final.json'
    if args.dry_run:
        print(f'\n--dry-run, would write {out}')
        return
    if not args.preprocessed.is_dir():
        raise SystemExit(f'{args.preprocessed} does not exist -- run '
                         'nnUNetv2_plan_and_preprocess first')
    out.write_text(json.dumps(splits, indent=2))
    print(f'\nwrote {out}')


if __name__ == '__main__':
    main()
