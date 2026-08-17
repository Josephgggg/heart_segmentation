import logging
import os
import numpy as np
import torch
from PIL import Image
from functools import lru_cache
from functools import partial
from itertools import repeat
from multiprocessing import Pool
from os import listdir
from os.path import splitext, isfile, join
from pathlib import Path
from torch.utils.data import Dataset
from tqdm import tqdm
import tifffile
import cv2
import pydicom
from pydicom.pixel_data_handlers.util import apply_modality_lut
import nibabel as nib
from collections import OrderedDict
from scipy.ndimage import gaussian_filter, map_coordinates
import random
import torch.utils.data as tud

def load_image(filename):
    ext = splitext(filename)[1].lower()
    if ext == '.dcm':
        ds = pydicom.dcmread(filename)
        img = apply_modality_lut(ds.pixel_array, ds).astype(np.float32)
        if img.ndim == 4:
            img = img[..., 0]
        if getattr(ds, 'PhotometricInterpretation', '') == 'MONOCHROME1':
            img = img.max() - img
        return img
    elif ext in ['.tif', '.tiff']:
        return tifffile.imread(filename)
    elif ext == '.npy':
        return np.load(filename)
    elif ext in ['.pt', '.pth']:
        return torch.load(filename).numpy()
    else:
        return np.asarray(Image.open(filename))


def _atomic_npy_save(path, arr):
    """np.save(path, arr), but via a per-process temp file + os.replace.

    Multiple processes can end up racing to populate the same disk cache
    entry (e.g. two notebooks sharing preprocessed_cache/ while it's still
    partially warm). A plain np.save(path, ...) is not atomic -- a reader
    can observe a partially-written file mid-save. Writing to a uniquely
    named temp file in the same directory and renaming into place sidesteps
    that: os.replace is atomic on the same filesystem, so any reader either
    sees the old complete file or the new complete file, never a torn one.
    Redundant work from the race is still possible (last writer wins), just
    not corruption.
    """
    path = Path(path)
    tmp_path = path.with_name(f'{path.name}.tmp{os.getpid()}')
    with open(tmp_path, 'wb') as f:
        np.save(f, arr)
    os.replace(tmp_path, path)


def unique_mask_values(idx, mask_dir, mask_suffix):
    mask_file = list(mask_dir.glob(idx + mask_suffix + '.*'))[0]
    mask = np.asarray(load_image(mask_file))
    if mask.ndim == 2:
        return np.unique(mask)
    elif mask.ndim == 3:
        mask = mask.reshape(-1, mask.shape[-1])
        return np.unique(mask, axis=0)
    else:
        raise ValueError(f'Loaded masks should have 2 or 3 dimensions, found {mask.ndim}')


class BasicDataset(Dataset):
    def __init__(self, images_dir: str, mask_dir: str, scale: float = 1.0, mask_suffix: str = ''):
        self.images_dir = Path(images_dir)
        self.mask_dir = Path(mask_dir)
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale
        self.mask_suffix = mask_suffix

        self.ids = [splitext(file)[0] for file in listdir(images_dir) if isfile(join(images_dir, file)) and not file.startswith('.')]
        if not self.ids:
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')

        logging.info(f'Creating dataset with {len(self.ids)} examples')
        logging.info('Scanning mask files to determine unique values')
        with Pool() as p:
            unique = list(tqdm(
                p.imap(partial(unique_mask_values, mask_dir=self.mask_dir, mask_suffix=self.mask_suffix), self.ids),
                total=len(self.ids)
            ))

        self.mask_values = list(sorted(np.unique(np.concatenate(unique), axis=0).tolist()))
        logging.info(f'Unique mask values: {self.mask_values}')

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def preprocess(mask_values, img, scale, is_mask):
        w, h = img.shape[:2]
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
        interp = cv2.INTER_NEAREST if is_mask else cv2.INTER_CUBIC
        img = cv2.resize(img, (newW, newH), interpolation=interp)

        if is_mask:
            mask = np.zeros((newH, newW), dtype=np.int64)
            for i, v in enumerate(mask_values):
                mask[img == v] = i
            return mask
        else:
            img = img.astype(np.float32)
            # per-slice robust min-max normalization (handles arbitrary 32-bit intensity range)
            lo, hi = np.percentile(img, [0.5, 99.5])
            img = np.clip(img, lo, hi)
            img = (img - lo) / (hi - lo + 1e-8)
            img = img[np.newaxis, ...]          # (H, W) -> (1, H, W), single-channel MRI
            return img

    def __getitem__(self, idx):
        name = self.ids[idx]
        mask_file = list(self.mask_dir.glob(name + self.mask_suffix + '.*'))
        img_file = list(self.images_dir.glob(name + '.*'))

        assert len(img_file) == 1, f'Either no image or multiple images found for the ID {name}: {img_file}'
        assert len(mask_file) == 1, f'Either no mask or multiple masks found for the ID {name}: {mask_file}'
        mask = load_image(mask_file[0])
        img = load_image(img_file[0])

        assert img.shape[:2] == mask.shape[:2], \
            f'Image and mask {name} should be the same size, but are {img.shape} and {mask.shape}'

        img = self.preprocess(self.mask_values, img, self.scale, is_mask=False)
        mask = self.preprocess(self.mask_values, mask, self.scale, is_mask=True)

        return {
            'image': torch.as_tensor(img.copy()).float().contiguous(),
            'mask': torch.as_tensor(mask.copy()).long().contiguous()
        }


# class CarvanaDataset(BasicDataset):
#     def __init__(self, images_dir, mask_dir, scale=1):
#         super().__init__(images_dir, mask_dir, scale, mask_suffix='_mask')

class VolumeMRIDataset(Dataset):
    # water keeps the 4 chambers (label 5/EAT falls through to background);
    # fat keeps only EAT (labels 1-4 fall through to background instead);
    # both stacks water+fat as 2 channels and keeps every label (0-5) intact.
    PHASE_MASK_VALUES = {
        'water': [0.0, 1.0, 2.0, 3.0, 4.0],
        'fat':   [0.0, 5.0],
        'both':  [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
    }

    def __init__(self, images_dir, mask_dir, scale: float = 1.0, cache_size: int = 4700, disk_cache_dir=None,
                 phase: str = 'water'):
        if phase not in self.PHASE_MASK_VALUES:
            raise ValueError(f"phase must be one of {sorted(self.PHASE_MASK_VALUES)}, got {phase!r}")
        self.images_dir = Path(images_dir)
        self.mask_dir = Path(mask_dir)
        self.scale = scale
        self.cache_size = cache_size
        # Each patient's .dcm holds both phases back to back: the first half of
        # slices is the water acquisition, the second half is fat, both sharing
        # the same NIfTI mask geometry. 'phase' picks which half of the volume
        # gets loaded and which mask label survives (see _read_and_process_volume).
        self.phase = phase

        # processed volumes get saved/loaded on disk
        self.disk_cache_dir = Path(disk_cache_dir) if disk_cache_dir else self.images_dir.parent / 'preprocessed_cache'
        self.disk_cache_dir.mkdir(parents=True, exist_ok=True)

        self._volume_cache = OrderedDict() # patient_id -> (img_vol, mask_vol)

        self.patient_files = sorted(self.images_dir.glob('*.dcm'))
        assert self.patient_files, f'No .dcm files found in {images_dir}'

        self.mask_file_for = {}
        self.n_slices_for = {}
        self.index = []   # list of (patient_id, slice_idx)

        for img_path in self.patient_files:
            patient_id = img_path.stem
            mask_matches = list(self.mask_dir.glob(patient_id + '.nii*'))  # matches .nii or .nii.gz
            assert len(mask_matches) == 1, f'Expected 1 mask for {patient_id}, found {mask_matches}'
            self.mask_file_for[patient_id] = mask_matches[0]

            # NEW: if a disk cache already exists, read the slice count from THAT (fast),
            # instead of opening the raw .dcm just to check its frame count
            img_cache_path, mask_cache_path = self._disk_cache_paths(patient_id)
            if img_cache_path.exists():
                n = np.load(img_cache_path, mmap_mode='r').shape[0]
            else:
                ds = pydicom.dcmread(img_path)
                n_total = ds.pixel_array.shape[0] if ds.pixel_array.ndim in (3, 4) else 1
                n = n_total // 2

            self.index.extend((patient_id, s) for s in range(n))

        logging.info(f'Found {len(self.patient_files)} patients, {len(self.index)} total slices')
        logging.info('Scanning mask files to determine unique values...')
        #all_values = set()
        #for patient_id in self.mask_file_for:
        #    _, mask_vol = self._get_volume(patient_id)   # uses disk cache if available
        #    all_values.update(np.unique(mask_vol).tolist())
        #self.mask_values = sorted(all_values)
        self.mask_values = self.PHASE_MASK_VALUES[phase]
        logging.info(f'Unique mask values: {self.mask_values}')

    def _disk_cache_paths(self, patient_id):
        # water keeps the original unsuffixed filenames so pre-existing caches stay
        # valid; only fat (the new phase) gets a suffix, so the two never collide.
        suffix = '' if self.phase == 'water' else f'_{self.phase}'
        return (self.disk_cache_dir / f'{patient_id}{suffix}_img.npy',
                self.disk_cache_dir / f'{patient_id}{suffix}_mask.npy')

    def spacing_for(self, patient_id):
        """Voxel spacing in mm as (slice, row, col), accounting for self.scale.

        Read lazily from the DICOM header only (stop_before_pixels), so this stays
        cheap even though the pixel data usually comes from the .npy disk cache.
        Spacing is NOT uniform across this dataset, so never hardcode it.
        """
        if not hasattr(self, '_spacing_cache'):
            self._spacing_cache = {}
        if patient_id not in self._spacing_cache:
            ds = pydicom.dcmread(self.images_dir / f'{patient_id}.dcm', stop_before_pixels=True)
            row_mm, col_mm = (float(v) for v in ds.PixelSpacing)
            slice_mm = float(getattr(ds, 'SpacingBetweenSlices', None)
                             or getattr(ds, 'SliceThickness', 1.0))
            # preprocess() resizes in-plane by self.scale; slice axis is untouched
            self._spacing_cache[patient_id] = (slice_mm, row_mm / self.scale, col_mm / self.scale)
        return self._spacing_cache[patient_id]

    def __len__(self):
        return len(self.index)

    def _read_and_process_volume(self, patient_id):
        img_cache_path, mask_cache_path = self._disk_cache_paths(patient_id)

        # NEW: fast path — already preprocessed, just load it
        if img_cache_path.exists() and mask_cache_path.exists():
            img_vol = np.load(img_cache_path)
            mask_vol = np.load(mask_cache_path)
            return img_vol, mask_vol

        # slow path — first time seeing this patient, do the real work
        img_path = self.images_dir / f'{patient_id}.dcm'
        ds = pydicom.dcmread(img_path)
        img_vol = apply_modality_lut(ds.pixel_array, ds).astype(np.float32)
        if img_vol.ndim == 4:
            img_vol = img_vol[..., 0]
        if getattr(ds, 'PhotometricInterpretation', '') == 'MONOCHROME1':
            img_vol = img_vol.max() - img_vol
        if img_vol.ndim == 2:
            img_vol = img_vol[np.newaxis, ...]

        # Each volume is water slices followed by fat slices. Anchor the fat half
        # at the END rather than slicing from the midpoint, so both phases yield
        # exactly `half` slices even when n_total is odd (midpoint slicing would
        # give the fat half one extra slice and desync it from the mask/index
        # count computed in __init__).
        n_total = img_vol.shape[0]
        half = n_total // 2
        water_vol, fat_vol = img_vol[:half], img_vol[n_total - half:]
        if self.phase == 'water':
            img_vol = water_vol
        elif self.phase == 'fat':
            img_vol = fat_vol
        else:  # 'both' -- (half, 2, H, W): channel 0 = water, channel 1 = fat
            img_vol = np.stack([water_vol, fat_vol], axis=1)

        mask_vol = nib.load(self.mask_file_for[patient_id]).get_fdata()
        mask_vol = np.transpose(mask_vol, (2, 0, 1))
        mask_vol = np.rot90(mask_vol[:, :, ::-1], k=1, axes=(1, 2)).astype(np.float32)

        FAT_LABEL = 5.0
        if self.phase == 'water':
            mask_vol[mask_vol == FAT_LABEL] = 0.0   # drop EAT, keep the 4 chambers
        elif self.phase == 'fat':
            mask_vol[mask_vol != FAT_LABEL] = 0.0   # drop the chambers, keep only EAT
        # 'both': mask_vol already carries labels 0-5 from the NIfTI as-is -- no relabelling needed

        assert img_vol.shape[0] == mask_vol.shape[0], \
            f'{patient_id}: {img_vol.shape[0]} image slices vs {mask_vol.shape[0]} mask slices — mismatch'

        # NEW: save the result to disk so future runs skip all of the above.
        # Atomic (temp file + rename) -- see _atomic_npy_save -- since another
        # process (e.g. a second notebook sharing this cache) may be writing
        # the same patient concurrently.
        logging.info(f'Preprocessing {patient_id} for the first time, saving to disk cache...')
        _atomic_npy_save(img_cache_path, img_vol)
        _atomic_npy_save(mask_cache_path, mask_vol)

        return img_vol, mask_vol
    
    def _get_volume(self, patient_id):
        #worker_info = tud.get_worker_info()
        #worker_id = worker_info.id if worker_info else 'main'
        if patient_id in self._volume_cache:
            self._volume_cache.move_to_end(patient_id)
            return self._volume_cache[patient_id]

        #print(f'[worker {worker_id}] CACHE MISS — loading {patient_id}')
        img_vol, mask_vol = self._read_and_process_volume(patient_id)
        self._volume_cache[patient_id] = (img_vol, mask_vol)
        self._volume_cache.move_to_end(patient_id)
        if len(self._volume_cache) > self.cache_size:
            self._volume_cache.popitem(last=False)
        return img_vol, mask_vol

    def __getitem__(self, idx):
        patient_id, slice_idx = self.index[idx]
        img_vol, mask_vol = self._get_volume(patient_id)   # uses the cache now
        img, mask = img_vol[slice_idx], mask_vol[slice_idx]

        if self.phase == 'both':
            # img is (2, H, W): water, fat. Normalize each channel independently --
            # preprocess() computes its 0.5/99.5 percentile clip over whatever 2D
            # array it's given, so combining channels before clipping would let one
            # channel's brightness distribution skew the other's normalization.
            img = np.concatenate([
                BasicDataset.preprocess(self.mask_values, img[0], self.scale, is_mask=False),
                BasicDataset.preprocess(self.mask_values, img[1], self.scale, is_mask=False),
            ], axis=0)
        else:
            img = BasicDataset.preprocess(self.mask_values, img, self.scale, is_mask=False)
        mask = BasicDataset.preprocess(self.mask_values, mask, self.scale, is_mask=True)

        return {
            'image': torch.as_tensor(img.copy()).float().contiguous(),
            'mask': torch.as_tensor(mask.copy()).long().contiguous(),
            'patient_id': patient_id,
            'slice_idx': slice_idx
        }


class AugmentedDataset(Dataset):
    """Wraps a dataset (typically a Subset of VolumeMRIDataset) and applies
    random rotation + elastic deformation to each (image, mask) pair, identically.
    Only wrap your TRAINING subset with this — never validation."""

    def __init__(self, base_dataset, rotate_prob=0.5, rotate_range=20,
                 elastic_prob=0.3, elastic_alpha=20, elastic_sigma=4,
                 contrast_prob=0.3, contrast_range=(0.4, 1.6),
                 noise_prob=0.3, noise_std=0.1):
        self.base_dataset = base_dataset
        self.rotate_prob = rotate_prob
        self.rotate_range = rotate_range      # max degrees, either direction
        self.elastic_prob = elastic_prob
        self.elastic_alpha = elastic_alpha    # deformation strength
        self.elastic_sigma = elastic_sigma    # smoothness of the deformation
        self.contrast_prob = contrast_prob
        self.contrast_range = contrast_range      # (min_factor, max_factor)
        self.noise_prob = noise_prob
        self.noise_std = noise_std                # std dev of gaussian noise, in normalized [0,1] intensity units

    def __len__(self):
        return len(self.base_dataset)

    def _rotate(self, img, mask):
        angle = random.uniform(-self.rotate_range, self.rotate_range)
        h, w = img.shape[-2:]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)

        # Same rotation matrix applied to every channel -- channels are spatially
        # co-registered (e.g. water+fat) and must warp identically.
        img_rot = np.stack([
            cv2.warpAffine(img[c], M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
            for c in range(img.shape[0])
        ], axis=0).astype(np.float32)
        mask_rot = cv2.warpAffine(mask.astype(np.float32), M, (w, h),
                                   flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

        return img_rot, mask_rot

    def _elastic_deform(self, img, mask):
        h, w = img.shape[-2:]
        dx = gaussian_filter(np.random.rand(h, w) * 2 - 1, self.elastic_sigma) * self.elastic_alpha
        dy = gaussian_filter(np.random.rand(h, w) * 2 - 1, self.elastic_sigma) * self.elastic_alpha

        x, y = np.meshgrid(np.arange(w), np.arange(h))
        coords = np.array([(y + dy).ravel(), (x + dx).ravel()])   # (row, col) order for map_coordinates

        # Same displacement field applied to every channel, for the same reason as _rotate.
        img_def = np.stack([
            map_coordinates(img[c], coords, order=3, mode='reflect').reshape(h, w)
            for c in range(img.shape[0])
        ], axis=0).astype(np.float32)
        mask_def = map_coordinates(mask.astype(np.float32), coords, order=0, mode='reflect').reshape(h, w)

        return img_def, mask_def

    def _adjust_contrast(self, img):
        factor = random.uniform(*self.contrast_range)
        mean = img.mean()
        img = (img - mean) * factor + mean
        img = np.clip(img, 0.0, 1.0)
        return img.astype(np.float32)

    def _add_intensity_noise(self, img):
        noise = np.random.normal(loc=0.0, scale=self.noise_std, size=img.shape).astype(np.float32)
        img = img + noise
        img = np.clip(img, 0.0, 1.0)
        return img.astype(np.float32)

    def __getitem__(self, idx):
        sample = self.base_dataset[idx]
        img = sample['image'].numpy()   # (1, H, W) float32
        mask = sample['mask'].numpy()   # (H, W) int64

        if random.random() < self.rotate_prob:
            img, mask = self._rotate(img, mask)

        if random.random() < self.elastic_prob:
            img, mask = self._elastic_deform(img, mask)

        if random.random() < self.contrast_prob:
            img = self._adjust_contrast(img)

        if random.random() < self.noise_prob:
            img = self._add_intensity_noise(img)

        return {
            'image': torch.as_tensor(img.copy()).float().contiguous(),
            'mask': torch.as_tensor(mask.copy()).long().contiguous()
        }


# class MRIDataset(BasicDataset):
#     def __init__(self, images_dir, mask_dir, scale=1):
#         super().__init__(images_dir, mask_dir, scale, mask_suffix='')  # match your actual suffix