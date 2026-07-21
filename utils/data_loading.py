import logging
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
    def __init__(self, images_dir, mask_dir, scale: float = 1.0, cache_size: int = 1000, disk_cache_dir=None):
        self.images_dir = Path(images_dir)
        self.mask_dir = Path(mask_dir)
        self.scale = scale
        self.cache_size = cache_size
        
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
        self.mask_values = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
        logging.info(f'Unique mask values: {self.mask_values}')

    def _disk_cache_paths(self, patient_id):
        return (self.disk_cache_dir / f'{patient_id}_img.npy',
                self.disk_cache_dir / f'{patient_id}_mask.npy')

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

        n_total = img_vol.shape[0]
        img_vol = img_vol[:n_total // 2]

        mask_vol = nib.load(self.mask_file_for[patient_id]).get_fdata()
        mask_vol = np.transpose(mask_vol, (2, 0, 1))
        mask_vol = np.rot90(mask_vol[:, :, ::-1], k=1, axes=(1, 2)).astype(np.float32)

        assert img_vol.shape[0] == mask_vol.shape[0], \
            f'{patient_id}: {img_vol.shape[0]} image slices vs {mask_vol.shape[0]} mask slices — mismatch'

        # NEW: save the result to disk so future runs skip all of the above
        logging.info(f'Preprocessing {patient_id} for the first time, saving to disk cache...')
        np.save(img_cache_path, img_vol)
        np.save(mask_cache_path, mask_vol)

        return img_vol, mask_vol
    
    def _get_volume(self, patient_id):
        if patient_id in self._volume_cache:
            self._volume_cache.move_to_end(patient_id)
            return self._volume_cache[patient_id]

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

        img = BasicDataset.preprocess(self.mask_values, img, self.scale, is_mask=False)
        mask = BasicDataset.preprocess(self.mask_values, mask, self.scale, is_mask=True)

        return {
            'image': torch.as_tensor(img.copy()).float().contiguous(),
            'mask': torch.as_tensor(mask.copy()).long().contiguous(),
            'patient_id': patient_id,
            'slice_idx': slice_idx
        }



# class MRIDataset(BasicDataset):
#     def __init__(self, images_dir, mask_dir, scale=1):
#         super().__init__(images_dir, mask_dir, scale, mask_suffix='')  # match your actual suffix