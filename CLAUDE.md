# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

This is a fork of [milesial/Pytorch-UNet](https://github.com/milesial/Pytorch-UNet), repurposed for **cardiac MRI segmentation** rather than the original Carvana car-image dataset. The README, `hubconf.py`, and parts of `predict.py` still describe the original Carvana/RGB use case and are stale — the actual current pipeline (`train.py`, `utils/data_loading.py`) trains a single-channel (`n_channels=1`) UNet on multi-slice DICOM MRI volumes with NIfTI masks. When making changes, trust the code over the README.

## Commands

There is no test suite, linter, or build step in this repo. Development is done by running the scripts directly.

Install dependencies (note: `requirements.txt` is incomplete — it's missing `torch`, `torchvision`, `opencv-python`, `pydicom`, `nibabel`, `tifffile`, and `scipy`, all of which are imported by `utils/data_loading.py`; install these manually if not already present):
```bash
pip install -r requirements.txt
```

Train:
```bash
python train.py --amp                      # mixed precision, recommended
python train.py --epochs 40 --batch-size 8 --scale 1.0 --classes 5 --amp --run-name baseline
python train.py --validation 15 --test 15  # percent of PATIENTS for val / test
python train.py --loss tversky_ce --no-test-eval   # one arm of the loss ablation
```
Run `python train.py -h` for the full flag list (`--load` to resume from a `.pth` checkpoint, `--bilinear` for bilinear upsampling instead of transposed conv, `--augment` to enable training-set augmentation, `--split-seed` to change the patient shuffle, `--no-test-eval` to skip the final test scoring, `--loss` to pick the loss arm, `--select-on` to change the checkpoint-selection metric).

Predict:
```bash
python predict.py --model MODEL.pth --input image.dcm --output out.png
python predict.py -i img1 img2 --viz --no-save   # visualize without saving
```

Training runs log to Weights & Biases under the `cardiac-mri-segmentation` project (`wandb` must be configured/logged in, or it falls back to an anonymous run).

Docker (image built from `nvcr.io/nvidia/pytorch:22.11-py3`):
```bash
docker build -t unet .
```
The GitHub Actions workflow in `.github/workflows/main.yml` auto-publishes this image to Docker Hub/GHCR on push to `master` — it targets the upstream `milesial/unet` repo, not this fork, so it's effectively inert here.

## Architecture

**Model** (`unet/`): Standard UNet (`unet_model.py` + `unet_parts.py`) — `DoubleConv`/`Down`/`Up`/`OutConv` blocks, optional bilinear upsampling, `use_checkpointing()` for gradient checkpointing on OOM. Unmodified from upstream.

**Data loading** (`utils/data_loading.py`) has two dataset implementations:
- `BasicDataset` — the original upstream loader: one image file + one mask file per example, arbitrary format via `load_image()` (dicom/tiff/npy/pt/PIL-supported).
- `VolumeMRIDataset` — the one actually used for cardiac MRI. Each "patient" is a single `.dcm` file in `data/imgs/` containing a multi-slice 3D volume, paired with a NIfTI mask (`.nii`/`.nii.gz`) in `data/masks/` matched by filename stem. Individual slices become dataset items, indexed as `(patient_id, slice_idx)` pairs in `self.index`. Only the first half of slices per volume are used (`img_vol[:n_total // 2]`) and label `5.0` ("fat") is relabeled to background. Processed volumes are cached to disk as `.npy` under `data/preprocessed_cache/` (or a custom dir) so subsequent runs skip DICOM/NIfTI parsing, and an in-memory `OrderedDict` LRU cache (`cache_size`, default 4700) holds recently used volumes across `__getitem__` calls.

`train.py` tries `VolumeMRIDataset` first and falls back to `BasicDataset` on `AssertionError`/`RuntimeError`/`IndexError` — so a `data/` directory laid out for the wrong loader will fail silently into the wrong dataset type rather than erroring clearly. Check `dataset.index`/`dataset.mask_file_for` if training behaves unexpectedly.

`PatientGroupedSampler` (`train.py`) shuffles at the patient level rather than the slice level, so slices from the same patient stay adjacent within an epoch (all patients still get randomly ordered, and slices within a patient are shuffled too) — this matters if you're debugging batch composition or data leakage between train/val.

**Splitting is by patient, never by slice** (`split_patients` in `train.py`) — all slices of a patient land in exactly one of train/validation/test, so val and test patients are ones the model never trains on. Defaults are 15% val / 15% test, which on the current 49 patients gives 35/7/7 patients ≈ 71%/14%/15% of the 4654 slices.

Two properties of `split_patients` that matter:
- **The test block is taken from the front of the shuffled patient list**, so changing `val_percent` moves only the train/val boundary and leaves the test set identical. Changing `split_seed` reshuffles everything, including test.
- `split_seed` must stay fixed for the life of the project. Varying it across runs means the "held-out" test set has effectively been trained on.

The test set is scored exactly once, after training, using the **best-validation weights** (kept in RAM as `best_state` and reloaded before the final `evaluate`) — not the last epoch. Set `eval_test=False` / `--no-test-eval` to skip it. Treat the test number as a final report, not a signal to tune against.

Note: `CADRE_1116`, `CADRE_1382`, `CADRE_1395`, and `CADRE_1404` each have two scan files (`_first`/`_second`), so 49 files cover 45 distinct subjects. These are deliberately treated as **independent patients** for splitting — two scans of the same subject can land in different sets.

`train_model` returns a results dict: `loss` (the arm name), best/final val Dice, `best_val_macro_dice`, best epoch, `test_dice`, set sizes, the val and test patient lists, and per-epoch `history`.

`AugmentedDataset` wraps the training subset (never validation) and applies, per sample, optional random rotation, elastic deformation, contrast jitter, and Gaussian intensity noise — img/mask pairs are transformed identically for the geometric ops. Enable it via `augment=True` / `--augment`; it is **off** by default.

## Seeding

`set_seed(seed, deterministic=False)` seeds `random`, `numpy`, `torch` and CUDA. **Call it before constructing the model** — `UNet(...)` draws its initial weights from the torch RNG, so seeding only inside `train_model` leaves the init random. Both `train.py`'s `__main__` and the notebook do this. `--seed` / `seed=` varies model init, batch order and augmentation; `--split-seed` is separate and controls only the patient split (keep it fixed).

`seed_worker` is passed as the DataLoader `worker_init_fn`, and the train loader gets an explicit seeded `generator`. This is not cosmetic — on fork the two non-torch RNGs misbehave in opposite ways:

- **numpy is inherited**, so every worker replays the *same* `np.random` stream. In `AugmentedDataset` that covers the elastic deformation fields and the Gaussian noise, so with `num_workers=4` there were ~4× fewer distinct warp fields and noise patterns than intended.
- **Python's `random` is auto-reseeded from OS entropy** (CPython registers `os.register_at_fork(after_in_child=seed)`), so rotation angles and contrast factors were never reproducible run-to-run.

`seed_worker` seeds both from `torch.initial_seed()` (= `base_seed + worker_id`, distinct per worker), fixing duplication and irreproducibility together. Runs before this change used `augment=False` and are unaffected; older augmented runs (`20_epoch_data_aug`, `data_aug_20_epoch`) predate the fix.

`deterministic=True` additionally restricts cuDNN to reproducible conv algorithms, which costs speed; the seeds alone remove the large variance sources, so it is off by default.

**Preprocessing** (`BasicDataset.preprocess`, shared by both dataset classes): images are per-slice robust-normalized using the 0.5th/99.5th percentile (not global min/max) before scaling; masks are resized with nearest-neighbor and re-mapped through `mask_values` to contiguous class indices.

**Training loop** (`train.py`): RMSprop optimizer, `ReduceLROnPlateau` scheduler keyed on validation Dice, AMP via `torch.autocast`/`GradScaler`, gradient clipping (`clip_grad_norm_`), and a loss selected from `utils/losses.py` (see **Losses** below; the loss is computed *outside* the autocast block, in fp32). Validation runs **both** mid-epoch (every `n_train // (5 * batch_size)` steps, ≈5×/epoch) and once at epoch end. `scheduler.step()` is called *only* in the mid-epoch block, so `ReduceLROnPlateau(patience=5)` is counted in mid-epoch evaluations (≈1 epoch), not in epochs — with the default `factor=0.1` the LR drops 10× after roughly one stagnant epoch. This is deliberate and matches the runs that produced the existing checkpoints; if you move `scheduler.step()` to epoch end, raise `patience` to keep the LR schedule comparable. The mid-epoch block also logs weight/gradient histograms for every parameter, which is the bulk of its cost. Checkpoints save to `checkpoints/<run_name>/checkpoint_epoch<N>.pth`, where `run_name` defaults to a timestamp; a `best.pth` copy of the best epoch is also written. **Best is chosen on per-patient macro Dice** (`metrics.macro_dice`, whole-volume, foreground classes), not on `evaluate.py`'s slice-wise Dice — `--select-on slice_dice` restores the old behaviour. The slice-wise number scores an absent class as 1.0, so selecting on it rewards predicting nothing on empty slices; that biases any comparison between losses that differ in empty-class handling. Both numbers are logged per epoch (`val_dice`, `val_macro_dice` in `history`). The state dict has `mask_values` injected into it, so `predict.py`/reload code must `pop`/`del` that key before calling `load_state_dict`.

There's dead/commented-out code throughout `train.py`'s training loop (debug prints, an early-return dict, a disabled non-finite-loss skip) — treat as scratch history, not something to preserve when editing nearby.

## Losses

`utils/losses.py` holds every loss and the `LOSS_REGISTRY` that names them. Select with `--loss <key>` or `train_model(loss='<key>')`; `loss=` also accepts an already-built loss object. The chosen arm is recorded in `run_config.json` and in the returned dict.

**Hyperparameters live in the registry, not on the CLI**, so an arm is fully identified by its key and nothing about it has to be reconstructed later. To sweep a hyperparameter, add another key rather than another flag.

Losses are objects with `__call__(logits, targets)` and an optional `on_epoch_start(epoch, total_epochs)`; `train_model` calls the hook once per epoch, which is what drives the boundary arm's weight ramp. `Compound` takes `(loss, weight)` pairs where a weight is either a constant or a `(start, end)` tuple ramped linearly across epochs.

Arms: `ce`, `dice`, `dice_ce` (default), `dice_ce_legacy`, `tversky_ce`, `focal_tversky_ce`, `dice_ce_boundary`.

`DiceLoss` differs from the upstream `utils/dice_score.dice_loss` in three ways that all matter:

- **background excluded** — CE handles class 0; pooling it into Dice lets background agreement dominate the score.
- **reduced per class, then averaged** — otherwise LV's pixel count swamps the atria.
- **numerator/denominator aggregated over the batch** ("batch Dice", as nnU-Net does for 2D). With per-slice Dice an absent class scores `eps/eps ≈ 1.0` with no gradient, so a false positive on an empty slice costs almost nothing — and 33.5% of (slice, class) pairs here are empty.

Measured on synthetic volumes, the corrected loss penalises a confident false positive on an empty class ~11× harder than the pooled version (0.167 vs 0.0156), and penalises dropping a small structure entirely ~64× harder (0.25 vs 0.0039). `dice_ce_legacy` reproduces the old pooled behaviour so pre-fix checkpoints stay interpretable; `utils/dice_score.py` itself is unchanged because `evaluate.py` still uses its coefficient.

`BoundaryLoss` (Kervadec 2019) computes signed distance maps on CPU per batch via `scipy.ndimage.distance_transform_edt` — φ is in **pixels**, not mm. Classes absent from a slice get φ = 0, contributing nothing rather than an undefined distance. It has no region anchor and is unstable alone; only use it compounded, as `dice_ce_boundary` does.

## Class labels

`CLASS_NAMES = {1: 'LV', 2: 'RV', 3: 'LA', 4: 'RA'}` in `train.py` (0 = background; label 5 = fat, relabelled to background by `VolumeMRIDataset`). The model is trained with `n_classes=5`.

This mapping was confirmed against the masks' own geometry, not assumed: inter-chamber contact areas reproduce the expected pattern — 1–2 is by far the largest interface (interventricular septum), 1–3 and 2–4 are the mitral and tricuspid valve planes, 3–4 is the interatrial septum, 1–4 is the small crux contact, and **2–3 is exactly zero**, which only RV/LA can be. Any relabelling proposal should be checked the same way before being accepted.

## Metrics and reporting

Two separate paths, deliberately:

- `evaluate.py` — slice-wise mean Dice, used as the in-training signal for the LR scheduler (only; checkpoint selection moved to `metrics.macro_dice`). **Not reportable.** `dice_coeff` scores a class that is absent from a slice as 1.0 (`eps/eps`), and 33.5% of (slice, class) pairs in this dataset are empty, so this number is inflated.
- `utils/metrics.py` — per-patient, per-class, 3D metrics for anything you actually report. Dice is computed on whole volumes (all four structures appear in 100% of volumes, so the empty-class problem disappears), and HD95 is in **mm**, using the medpy/ACDC convention (95th percentile of pooled bidirectional surface distances) so it is comparable with published numbers.

`metrics.report(net, dataset, indices, device, n_classes, out_dir=..., split_name=...)` scores a split and writes `<split>_metrics_per_patient.csv` (one row per patient × class) and `<split>_metrics_summary.csv` (mean/SD/median/IQR per class) into the run's checkpoint directory. `train_model` calls this automatically after training, using best-validation weights: **val is always scored, test only when `eval_test=True`**. That asymmetry is what lets an ablation (`eval_test=False`) produce comparable CSVs without touching the test set.

`metrics.macro_dice(net, dataset, indices, ...)` is the cheap variant used for per-epoch checkpoint selection — same per-patient volume Dice, but with `compute_hd95=False` so it skips the surface distances and the DICOM header read they need.

Five metrics per (patient, class), listed in `metrics.DEFAULT_METRICS`: `dice`, `precision`, `recall`, `hd95_mm`, `assd_mm`.

`precision`/`recall` are diagnostic, not endpoints. Dice is their harmonic mean, so it cannot separate over-segmentation from under-segmentation — low recall means the model is missing true voxels, low precision means it is claiming voxels it shouldn't. This is also the only direct readout of whether an asymmetric loss did what it was designed to do (`tversky_ce` uses β>α specifically to raise recall). nan conventions match `dice_binary`: recall is undefined with no ground truth, precision with no prediction — so a total miss gives recall 0.0 (real, counts) and precision nan. `compare_runs` deliberately does **not** include them by default: they would grow the Holm family from 12 tests to 20 and weaken the correction on the metrics being compared.

`summarise()` and `format_summary()` silently drop metrics absent from the rows they're given, so CSVs written before a metric existed (the pre-ASSD, pre-precision/recall runs) still load instead of raising `KeyError`.

HD95 and ASSD come from `surface_metrics()`, which computes both from one shared pair of distance transforms; both follow medpy conventions (HD95 = 95th percentile of *pooled* bidirectional distances; ASSD = mean of the two *directional means* — these are different poolings, deliberately). The per-patient CSV also stores `gt_voxels`/`pred_voxels`/`voxel_ml`, which is what makes Bland–Altman volume agreement derivable without re-running the model.

The summary CSV has a macro row (`class_name == 'all_classes_macro'`, `cls == -1`) carrying `hd95_mm_*` and `assd_mm_*` like every other row; the per-patient CSV deliberately has no macro row.

Each run directory also gets `run_config.json` — hyperparameters, split seed, the three patient lists, torch/GPU versions, and (after training) best epoch and per-epoch history. Without it a directory of CSVs from a dozen experiments is unattributable.

`metrics.compare_runs(csv_a, csv_b)` pairs two runs on `(patient_id, class_name)` and runs paired Wilcoxon signed-rank tests. **Both runs must share `split_seed`** or there are no shared patients to pair. Note the hard sample-size limit: the exact two-sided Wilcoxon floor is `2/2**n`, which is 0.0156 at n=7 test patients — so the 12 per-chamber × per-metric tests can never survive Holm correction (0.0156 × 12 = 0.19). `compare_runs` therefore Holm-corrects the per-chamber family only and leaves the macro row uncorrected as a single pre-specified primary endpoint. Per-chamber results at this n are descriptive, not confirmatory.

**The unit of analysis is the patient, not the slice.** Slices within a patient are highly correlated, so an SD over slices is misleadingly small; `summarise()` therefore aggregates each patient to one value per class first, then takes mean/SD across patients (sample SD, `ddof=1`). Report as `mean ± SD (n=patients)`.

Absent structures yield `nan` rather than a fake score, and are excluded from means — `summarise()` reports `n` and `n_missed` (ground truth present but prediction empty) alongside every statistic so a mean resting on few patients is visible. HD95 is undefined for an empty prediction, so a model that fails completely on a structure would otherwise appear to *improve* its mean HD95; `n_missed` is what catches that.

Voxel spacing is **not uniform** across this dataset (0.6968 / 0.6944 / 0.6481 mm in-plane) — never hardcode it. `VolumeMRIDataset.spacing_for(patient_id)` reads it lazily from the DICOM header (`stop_before_pixels`, ~0.02 s/patient) and corrects the in-plane values for `self.scale`.

## Notebooks

`unet_applied.ipynb` is the iterative exploration/pipeline notebook, used for interactive experimentation alongside the `.py` scripts rather than as the canonical pipeline.

`loss_ablation.ipynb` drives the loss ablation: arms × seeds in a resumable loop, seed-averaged per-patient CSVs, paired comparison against `dice_ce`, and a gated final test cell. It is deliberately validation-only (`eval_test=False` on every run) — choosing a loss is model selection, so scoring test during it would burn the held-out set. Its `SPLIT_SEED` must not be changed.

## Data layout

- `data/imgs/*.dcm` — one multi-slice DICOM volume per patient.
- `data/masks/*.nii[.gz]` — matching NIfTI segmentation volume, filename stem must match the image's.
- `data/preprocessed_cache/` and `preprocessed_cache/` — disk-cached `.npy` arrays from `VolumeMRIDataset`; safe to delete to force reprocessing, but large files (a git-ignore/cleanup candidate if disk space is tight).
- `checkpoints/` — saved model weights, both loose `.pth` files from older runs and per-run subdirectories from newer ones.
