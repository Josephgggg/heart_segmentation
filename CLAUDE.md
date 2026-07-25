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
python train.py --epochs 20 --batch-size 4 --learning-rate 1e-5 --scale 0.5 --classes 2
```
Run `python train.py -h` for the full flag list (`--load` to resume from a `.pth` checkpoint, `--bilinear` for bilinear upsampling instead of transposed conv, `--validation` for the val split percentage).

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

Train/val split (`train_model` in `train.py`) is done **by patient**, not by slice: patients are shuffled with a fixed seed (`random.Random(0)`), and a percentage of *patients* (`val_percent`) becomes the validation set. This avoids slices from the same patient leaking across the split.

`AugmentedDataset` wraps a training subset (never wrap validation) and applies, per sample, optional random rotation, elastic deformation, contrast jitter, and Gaussian intensity noise — img/mask pairs are transformed identically for the geometric ops. It's currently *not* wired into `train_model` (the wrapping line is commented out) — augmentation exists in the code but isn't active in the default training path.

**Preprocessing** (`BasicDataset.preprocess`, shared by both dataset classes): images are per-slice robust-normalized using the 0.5th/99.5th percentile (not global min/max) before scaling; masks are resized with nearest-neighbor and re-mapped through `mask_values` to contiguous class indices.

**Training loop** (`train.py`): RMSprop optimizer, `ReduceLROnPlateau` scheduler keyed on validation Dice, AMP via `torch.autocast`/`GradScaler`, gradient clipping (`clip_grad_norm_`), combined `CrossEntropyLoss`/`BCEWithLogitsLoss` + Dice loss (`utils/dice_score.py`). Validation runs both periodically mid-epoch (every `n_train // (5 * batch_size)` steps) and once at epoch end. Checkpoints save to `checkpoints/<run_name>/checkpoint_epoch<N>.pth`, where `run_name` defaults to a timestamp if not passed in; the state dict has `mask_values` injected into it, so `predict.py`/reload code must `pop`/`del` that key before calling `load_state_dict`.

There's dead/commented-out code throughout `train.py`'s training loop (debug prints, an early-return dict, a disabled non-finite-loss skip) — treat as scratch history, not something to preserve when editing nearby.

## Notebooks

`unet_applied.ipynb`, `unet_applied_v2.ipynb`, `unet_applied_v3.ipynb` are iterative exploration/pipeline notebooks (v3 is the most recent). They're used for interactive experimentation alongside the `.py` scripts rather than as the canonical pipeline.

## Data layout

- `data/imgs/*.dcm` — one multi-slice DICOM volume per patient.
- `data/masks/*.nii[.gz]` — matching NIfTI segmentation volume, filename stem must match the image's.
- `data/preprocessed_cache/` and `preprocessed_cache/` — disk-cached `.npy` arrays from `VolumeMRIDataset`; safe to delete to force reprocessing, but large files (a git-ignore/cleanup candidate if disk space is tight).
- `checkpoints/` — saved model weights, both loose `.pth` files from older runs and per-run subdirectories from newer ones.
