import argparse
import json
import logging
import os
import random
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from pathlib import Path
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from utils.data_loading import AugmentedDataset

import wandb
from evaluate import evaluate
from unet import UNet
from utils.data_loading import BasicDataset, VolumeMRIDataset
from utils.losses import LOSS_REGISTRY, build_loss, loss_name
from utils import metrics

import re
import numpy as np
from collections import defaultdict
from torch.utils.data import Subset
from torch.utils.data import Sampler
import datetime
import time

dir_img = Path('./data/imgs/')
dir_mask = Path('./data/masks/')
dir_checkpoint = Path('./checkpoints/')

# Cardiac chamber labels in the NIfTI masks. 0 = background. VolumeMRIDataset
# loads one (or both) of two phases from the same .dcm/.nii pair: 'water' keeps
# the 4 chambers (label 5/EAT relabelled to background), 'fat' keeps only EAT
# (labels 1-4 relabelled to background instead), 'both' stacks water+fat as a
# 2-channel input and keeps every label -- see its `phase` argument.
CLASS_NAMES = {1: 'LV', 2: 'RV', 3: 'LA', 4: 'RA'}
BOTH_CLASS_NAMES = {1: 'LV', 2: 'RV', 3: 'LA', 4: 'RA', 5: 'EAT'}   # 'EAT' matches
# PHASE_CLASS_NAMES['fat']'s class_name string, so metrics.compare_runs -- which
# pairs rows on (patient_id, class_name) -- can actually match this class
# against the existing dedicated fat-phase model's per-patient CSV.
PHASE_CLASS_NAMES = {
    'water': CLASS_NAMES,
    'fat': {1: 'EAT'},
    'both': BOTH_CLASS_NAMES,
}
# Model output channels (incl. background) appropriate for each phase, used as
# the --classes default when the user doesn't override it.
PHASE_N_CLASSES = {'water': 5, 'fat': 2, 'both': 6}
# Model input channels appropriate for each phase.
PHASE_N_CHANNELS = {'water': 1, 'fat': 1, 'both': 2}


class PatientGroupedSampler(Sampler):
    def __init__(self, index_subset):
        self.index_subset = index_subset

    def __iter__(self):
        by_patient = defaultdict(list)
        for local_i, (patient_id, _) in enumerate(self.index_subset):
            by_patient[patient_id].append(local_i)

        patients = list(by_patient.keys())
        random.shuffle(patients)

        order = []
        for p in patients:
            idxs = by_patient[p]
            random.shuffle(idxs)
            order.extend(idxs)
        return iter(order)

    def __len__(self):
        return len(self.index_subset)


def set_seed(seed: int = 0, deterministic: bool = False):
    """Seed every RNG that affects a training run.

    Call this BEFORE constructing the model -- weight initialisation consumes the
    torch RNG, so seeding only inside train_model leaves the initial weights
    random and two "identical" runs will still differ.

    `deterministic=True` additionally restricts cuDNN to reproducible conv
    algorithms. That costs speed, so it is off by default; the seeds above
    already remove the large sources of variance (init, batch order,
    augmentation), leaving only small floating-point non-associativity.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    logging.info(f'Seeded RNGs with {seed}' + (' (cuDNN deterministic)' if deterministic else ''))


def seed_worker(worker_id):
    """Give each DataLoader worker a distinct, reproducible numpy/random state.

    PyTorch reseeds `torch` per worker automatically. The other two RNGs behave
    differently on fork, and both are wrong for us:

      numpy  -- inherited from the parent, so every worker replays the SAME
                np.random stream. In AugmentedDataset that is the elastic
                deformation fields (np.random.rand) and the Gaussian noise
                (np.random.normal): with num_workers=4 there are roughly 4x
                fewer distinct warp fields and noise patterns than intended.
      random -- CPython registers os.register_at_fork(after_in_child=seed), so
                each worker is reseeded from OS entropy. That avoids duplication
                but makes the rotation angles and contrast factors
                irreproducible: identical settings give different augmentation
                every run.

    Seeding both from torch.initial_seed() (which is base_seed + worker_id, so
    distinct per worker) fixes duplication and irreproducibility together.
    """
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_dataset(img_scale: float = 0.5, phase: str = 'water'):
    try:
        return VolumeMRIDataset(dir_img, dir_mask, img_scale, phase=phase)
    except (AssertionError, RuntimeError, IndexError):
        return BasicDataset(dir_img, dir_mask, img_scale)


def split_patients(dataset, val_percent: float = 0.15, test_percent: float = 0.15, seed: int = 0):
    """Split patients three ways: train / validation / test.

    The split is by patient, never by slice, so every slice of a patient lands in
    exactly one set and the val/test patients are ones the model never trains on.

    The test block is taken from the front of the shuffled list, so changing
    val_percent moves only the train/val boundary and leaves the test set intact.
    Keep `seed` fixed for the life of the project or the test set stops being held out.

    Returns (train_idx, val_idx, test_idx) into dataset.index.
    """
    patients = sorted(dataset.mask_file_for)
    random.Random(seed).shuffle(patients)

    n = len(patients)
    n_test = max(1, round(n * test_percent)) if test_percent > 0 else 0
    n_val = max(1, round(n * val_percent)) if val_percent > 0 else 0
    assert n_test + n_val < n, \
        f'val_percent + test_percent leave no training patients ({n_val} + {n_test} of {n})'

    test_patients = set(patients[:n_test])
    val_patients = set(patients[n_test:n_test + n_val])

    by_patient = defaultdict(list)
    for i, (patient_id, _) in enumerate(dataset.index):
        by_patient[patient_id].append(i)

    test_idx = sorted(i for p in test_patients for i in by_patient[p])
    val_idx = sorted(i for p in val_patients for i in by_patient[p])
    train_idx = sorted(
        i for p, idxs in by_patient.items()
        if p not in test_patients and p not in val_patients
        for i in idxs
    )

    logging.info(
        f'Patient-level split of {n} patients / {len(dataset.index)} slices (seed {seed}):\n'
        f'        train: {n - n_val - n_test:>2} patients, {len(train_idx):>5} slices\n'
        f'        val:   {n_val:>2} patients, {len(val_idx):>5} slices  {sorted(val_patients)}\n'
        f'        test:  {n_test:>2} patients, {len(test_idx):>5} slices  {sorted(test_patients)}'
    )
    return train_idx, val_idx, test_idx


def _chunk_evenly(items, k):
    """Split `items` into k contiguous chunks, sizes differing by at most 1."""
    n = len(items)
    base, rem = divmod(n, k)
    chunks, start = [], 0
    for i in range(k):
        size = base + (1 if i < rem else 0)
        chunks.append(items[start:start + size])
        start += size
    return chunks


def split_patients_kfold(dataset, fold: int, k_folds: int = 5, test_percent: float = 0.15, seed: int = 0):
    """K-fold patient-level split of the non-test pool. `fold` (0-indexed) is validation.

    The test carve-out is the IDENTICAL computation split_patients uses -- same patient
    list, same seed, same shuffle, same n_test -- so the held-out test set does not
    change between a single-split run and a k-fold run at the same seed, and a test set
    already frozen by an existing single-split run stays valid. Only the remaining pool
    is treated differently: instead of one fixed val/train boundary, it is partitioned
    into k_folds near-equal chunks (in shuffled order) and fold `fold` is validation,
    the rest are train.

    Returns (train_idx, val_idx, test_idx) into dataset.index, same shape as split_patients.
    """
    assert k_folds >= 2, f'k_folds must be >= 2, got {k_folds}'
    assert 0 <= fold < k_folds, f'fold must be in [0, {k_folds}), got {fold}'

    patients = sorted(dataset.mask_file_for)
    random.Random(seed).shuffle(patients)

    n = len(patients)
    n_test = max(1, round(n * test_percent)) if test_percent > 0 else 0
    test_patients = set(patients[:n_test])
    pool = patients[n_test:]
    assert k_folds <= len(pool), \
        f'{k_folds} folds requested but only {len(pool)} patients remain after the {n_test}-patient test carve-out'

    folds = _chunk_evenly(pool, k_folds)
    val_patients = set(folds[fold])
    train_patients = set(pool) - val_patients

    by_patient = defaultdict(list)
    for i, (patient_id, _) in enumerate(dataset.index):
        by_patient[patient_id].append(i)

    test_idx = sorted(i for p in test_patients for i in by_patient[p])
    val_idx = sorted(i for p in val_patients for i in by_patient[p])
    train_idx = sorted(i for p in train_patients for i in by_patient[p])

    logging.info(
        f'Patient-level {k_folds}-fold split of {n} patients / {len(dataset.index)} slices '
        f'(seed {seed}), fold {fold}/{k_folds - 1}:\n'
        f'        train: {len(train_patients):>2} patients, {len(train_idx):>5} slices\n'
        f'        val:   {len(val_patients):>2} patients, {len(val_idx):>5} slices  {sorted(val_patients)}\n'
        f'        test:  {n_test:>2} patients, {len(test_idx):>5} slices  {sorted(test_patients)}'
    )
    return train_idx, val_idx, test_idx


def train_model(
        model,
        device,
        epochs: int = 5,
        batch_size: int = 1,
        learning_rate: float = 1e-5,
        val_percent: float = 0.15,
        test_percent: float = 0.15,
        save_checkpoint: bool = True,
        img_scale: float = 0.5,
        amp: bool = False,
        weight_decay: float = 1e-8,
        momentum: float = 0.9,
        gradient_clipping: float = 1.0,
        run_name=None,
        dataset=None,
        augment: bool = False,
        split_seed: int = 0,
        eval_test: bool = True,
        per_patient_metrics: bool = True,
        class_names=None,
        seed: int = 0,
        deterministic: bool = False,
        loss: str = 'dice_ce',
        select_on: str = 'macro_dice',
        lr_schedule: str = 'poly',
        phase: str = 'water',
        k_folds: int = 0,
        fold: int = 0,
):
    # 0. Reseed. NOTE: the caller built the model, so its weights were already
    # drawn -- call set_seed() before UNet(...) too for a bit-comparable run.
    set_seed(seed, deterministic)

    # Resolve the arm's name up front: the training loop below rebinds `loss` to
    # the per-batch tensor, so reading it afterwards would report "Tensor".
    loss_label = loss_name(loss)

    # 1. Create dataset
    if dataset is None:
        dataset = build_dataset(img_scale, phase=phase)

    # 2. Split into train / validation / test partitions, by patient.
    # k_folds > 0 replaces the single fixed val boundary with a k-fold split of the
    # non-test pool (see split_patients_kfold); val_percent is ignored in that case,
    # but the test carve-out is identical either way at the same split_seed.
    if k_folds > 0:
        train_idx, val_idx, test_idx = split_patients_kfold(
            dataset, fold=fold, k_folds=k_folds, test_percent=test_percent, seed=split_seed
        )
    else:
        train_idx, val_idx, test_idx = split_patients(
            dataset, val_percent=val_percent, test_percent=test_percent, seed=split_seed
        )

    train_set = Subset(dataset, train_idx)
    val_set = Subset(dataset, val_idx)
    test_set = Subset(dataset, test_idx)
    n_train, n_val, n_test = len(train_set), len(val_set), len(test_set)

    # 3. Create data loaders
    # augmentation wraps the TRAINING subset only — never validation or test
    if augment:
        train_set = AugmentedDataset(train_set)

    # `generator` makes the shuffle order reproducible; `worker_init_fn` gives each
    # worker its own numpy/random stream (see seed_worker -- this is also what
    # stops the four workers augmenting in lockstep).
    loader_generator = torch.Generator()
    loader_generator.manual_seed(seed)

    loader_args = dict(batch_size=batch_size, num_workers=4, pin_memory=True,
                       persistent_workers=True, worker_init_fn=seed_worker)
    train_loader = DataLoader(train_set, shuffle=True, generator=loader_generator, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=False, **loader_args)
    # the test loader is built once at the very end, so it gets no persistent workers

    if run_name is None:
        run_name = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    # water keeps unprefixed run names -- every existing checkpoint directory and
    # every notebook's already_done()/run_dir bookkeeping (loss_ablation.ipynb,
    # unet_applied.ipynb) was computed against these names before `phase` existed,
    # and those callers build run_dir themselves from the SAME string they pass in
    # here, so silently renaming it out from under them would desync the two and
    # make every finished run look unfinished. fat is new, so it gets the prefix.
    if phase != 'water':
        run_name = f'{phase}_{run_name}'
    run_checkpoint_dir = Path(dir_checkpoint) / run_name

    # Record exactly what produced this run, next to its metrics. Without this,
    # a directory of CSVs from a dozen experiments is unattributable later.
    run_config = {
        'run_name': run_name,
        'started': datetime.datetime.now().isoformat(timespec='seconds'),
        'epochs': epochs, 'batch_size': batch_size, 'learning_rate': learning_rate,
        'weight_decay': weight_decay, 'momentum': momentum,
        'gradient_clipping': gradient_clipping,
        'optimizer': 'RMSprop', 'lr_schedule': lr_schedule,
        'scheduler': {'poly': 'LambdaLR poly (1-e/E)**0.9, per epoch',
                      'cosine': 'CosineAnnealingLR, per epoch',
                      'plateau': 'ReduceLROnPlateau(max, patience=5), per epoch',
                      'constant': 'none'}[lr_schedule],
        'img_scale': img_scale, 'amp': amp, 'augment': augment,
        'loss': loss_label, 'select_on': select_on, 'phase': phase,
        'n_classes': model.n_classes, 'n_channels': model.n_channels,
        'bilinear': model.bilinear,
        'val_percent': val_percent, 'test_percent': test_percent, 'split_seed': split_seed,
        'k_folds': k_folds, 'fold': fold,
        'seed': seed, 'deterministic': deterministic,
        'n_train': n_train, 'n_val': n_val, 'n_test': n_test,
        'train_patients': sorted({dataset.index[i][0] for i in train_idx}),
        'val_patients': sorted({dataset.index[i][0] for i in val_idx}),
        'test_patients': sorted({dataset.index[i][0] for i in test_idx}),
        'device': str(device), 'torch_version': torch.__version__,
        'gpu': torch.cuda.get_device_name(0) if device.type == 'cuda' else None,
    }
    run_checkpoint_dir.mkdir(parents=True, exist_ok=True)
    with open(run_checkpoint_dir / 'run_config.json', 'w') as f:
        json.dump(run_config, f, indent=2)

    # (Initialize logging)
    experiment = wandb.init(
        project='cardiac-mri-segmentation',
        name=run_name,
        tags=[phase],
        reinit=True,
    )
    experiment.config.update(
        dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
             val_percent=val_percent, test_percent=test_percent, split_seed=split_seed,
             k_folds=k_folds, fold=fold,
             save_checkpoint=save_checkpoint, img_scale=img_scale, amp=amp, augment=augment,
             loss=loss_label, select_on=select_on, lr_schedule=lr_schedule, phase=phase,
             n_train=n_train, n_val=n_val, n_test=n_test)
    )

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}{f' (fold {fold}/{k_folds - 1})' if k_folds > 0 else ''}
        Test size:       {n_test} (held out, {'scored once after training' if eval_test else 'not scored'})
        Augmentation:    {augment}
        Loss:            {loss_label}
        LR schedule:     {lr_schedule}
        Select best on:  {select_on}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Images scaling:  {img_scale}
        Mixed Precision: {amp}
    ''')

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    optimizer = optim.RMSprop(model.parameters(),
                              lr=learning_rate, weight_decay=weight_decay, momentum=momentum, foreach=True)

    # LR schedule. 'poly' is nnU-Net's rule, lr0 * (1 - epoch/epochs)**0.9, stepped
    # once per epoch and depending on nothing but the epoch number.
    #
    # That last property is the reason it is the default here. ReduceLROnPlateau
    # fires off the validation curve, and the validation curve depends on the loss
    # being trained -- so in an ablation each arm would get its own LR trajectory
    # and "which loss is better" could not be separated from "which loss delayed
    # its LR drops". A deterministic schedule gives every arm the same trajectory.
    #
    # 'plateau' keeps the old behaviour available, but note it was stepped inside
    # the mid-epoch block (~5x/epoch), which made patience=5 mean roughly ONE
    # stagnant epoch; with factor=0.1 the LR fell 100x within 5 epochs and
    # training stopped by ~epoch 8. If you use it, step it once per epoch.
    if lr_schedule == 'poly':
        scheduler = optim.lr_scheduler.LambdaLR(
            optimizer, lambda e: (1 - e / max(1, epochs)) ** 0.9)
    elif lr_schedule == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    elif lr_schedule == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)
    elif lr_schedule == 'constant':
        scheduler = None
    else:
        raise ValueError(f'Unknown lr_schedule {lr_schedule!r}: '
                         "expected 'poly', 'cosine', 'plateau' or 'constant'")
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    # One ablation arm, resolved from its registry name (see utils/losses.py).
    criterion = build_loss(loss, model.n_classes)
    global_step = 0
    batch_skip_count = 0
    best_val_dice = 0.0
    best_val_macro_dice = float('nan')
    best_selection_score = -float('inf')
    best_epoch = 0
    best_state = None
    history = []

    # 5. Begin training
    for epoch in range(1, epochs + 1):
        model.train()
        # lets scheduled losses (the boundary arm) advance their weighting
        criterion.on_epoch_start(epoch, epochs)
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                #print("yo")
                images, true_masks = batch['image'], batch['mask']
                #print("yoyo")

                assert images.shape[1] == model.n_channels, \
                    f'Network has been defined with {model.n_channels} input channels, ' \
                    f'but loaded images have {images.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'
                
                images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                true_masks = true_masks.to(device=device, dtype=torch.long)

                with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                    masks_pred = model(images)

                # The loss is computed OUTSIDE autocast, in fp32. The reductions in
                # a Dice-family loss are the numerically fragile part, and under
                # autocast the .float() casts inside them do not reliably keep the
                # accumulation in fp32.
                loss = criterion(masks_pred, true_masks)

                # if not torch.isfinite(loss):
                #     batch_skip_count += 1
                #     logging.warning(f'Non-finite loss at step {global_step}, skipping batch.')
                #     print(f"Batch skip count: {batch_skip_count}")

                    # print("images:",
                    #     torch.isnan(images).any().item(),
                    #     torch.isinf(images).any().item())

                    # print("predictions:",
                    #     torch.isnan(masks_pred).any().item(),
                    #     torch.isinf(masks_pred).any().item())

                    # print("min/max:",
                    #     masks_pred.min().item(),
                    #     masks_pred.max().item())
                    
                    # return {
                    #     "images": images.detach().cpu(),
                    #     "true_masks": true_masks.detach().cpu(),
                    #     "pred_masks": masks_pred.detach().cpu(),
                    #     "patient_ids": batch["patient_id"],
                    #     "slice_idxs": batch["slice_idx"],
                    #     "loss": loss.detach().cpu(),
                    #     "pred_min": masks_pred.min().item(),
                    #     "pred_max": masks_pred.max().item(),
                    #     "pred_has_nan": torch.isnan(masks_pred).any().item(),
                    #     "pred_has_inf": torch.isinf(masks_pred).any().item(),
                    #     "image_min": images.min().item(),
                    #     "image_max": images.max().item(),
                    #     "mask_values": torch.unique(true_masks).tolist(),
                    # }
                    #return
                    # continue   # skips backward, optimizer step, and logging entirely for this batch

                #print("hello 1")
                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                grad_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                grad_scaler.step(optimizer)
                grad_scaler.update()
                #print("bye 1")

                #print("hello 2")
                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                experiment.log({
                    'train loss': loss.item(),
                    'step': global_step,
                    'epoch': epoch
                })
                pbar.set_postfix(**{'loss (batch)': loss.item()})
                #print("bye 2")

                # Evaluation round
                division_step = (n_train // (5 * batch_size))
                if division_step > 0:
                    if global_step % division_step == 0:
                        histograms = {}
                        for tag, value in model.named_parameters():
                            tag = tag.replace('/', '.')
                            if not (torch.isinf(value) | torch.isnan(value)).any():
                                histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())
                            if not (torch.isinf(value.grad) | torch.isnan(value.grad)).any():
                                histograms['Gradients/' + tag] = wandb.Histogram(value.grad.data.cpu())

                        # NOTE: scheduler.step() used to live here, so it ran ~5x
                        # per epoch and patience=5 meant one stagnant epoch. The
                        # step is now once per epoch, at the end of the loop below.
                        val_score = evaluate(model, val_loader, device, amp)

                        logging.info('Validation Dice score: {}'.format(val_score))
                        try:
                            experiment.log({
                                'learning rate': optimizer.param_groups[0]['lr'],
                                'validation Dice': val_score,
                                'images': wandb.Image(images[0].cpu()),
                                'masks': {
                                    'true': wandb.Image(true_masks[0].float().cpu()),
                                    'pred': wandb.Image(masks_pred.argmax(dim=1)[0].float().cpu()),
                                },
                                'step': global_step,
                                'epoch': epoch,
                                **histograms
                            })
                        except:
                            pass

        # End-of-epoch scoring. Two numbers, for two different jobs:
        #   epoch_val_dice       -- slice-wise, from evaluate.py. Kept only for
        #                           continuity with earlier runs. Inflated.
        #   epoch_val_macro_dice -- per-patient, per-class, whole-volume. This is
        #                           what selects the checkpoint.
        # Selecting on the slice-wise number would bias a loss ablation: it scores
        # an absent class as 1.0, so it rewards predicting nothing on empty slices,
        # and empty-class handling is precisely what the arms differ in.
        epoch_val_dice = float(evaluate(model, val_loader, device, amp))
        mean_epoch_loss = epoch_loss / len(train_loader)

        epoch_val_macro_dice = float('nan')
        if hasattr(dataset, 'index') and n_val > 0:
            epoch_val_macro_dice = metrics.macro_dice(
                model, dataset, val_idx, device, n_classes=model.n_classes,
                amp=amp, batch_size=batch_size,
            )

        selection_score = (epoch_val_dice if select_on == 'slice_dice'
                           else epoch_val_macro_dice)
        if not np.isfinite(selection_score):        # no val patients, or the cheap path failed
            selection_score = epoch_val_dice

        # LR is recorded BEFORE stepping, so history[i]['lr'] is the rate that
        # actually trained epoch i. Logging this per epoch is what makes an LR
        # collapse visible in run_config.json instead of only as a flat loss curve.
        epoch_lr = optimizer.param_groups[0]['lr']

        logging.info(f'Epoch {epoch}: lr = {epoch_lr:.3e}, mean train loss = {mean_epoch_loss:.4f}, '
                     f'val Dice (slice) = {epoch_val_dice:.4f}, '
                     f'val Dice (per-patient macro) = {epoch_val_macro_dice:.4f}')

        history.append({'epoch': epoch, 'lr': epoch_lr, 'train_loss': mean_epoch_loss,
                        'val_dice': epoch_val_dice,
                        'val_macro_dice': epoch_val_macro_dice})

        experiment.log({
            'epoch': epoch,
            'learning rate': epoch_lr,
            'epoch_train_loss': mean_epoch_loss,
            'epoch_val_dice': epoch_val_dice,
            'epoch_val_macro_dice': epoch_val_macro_dice,
        })

        # Step ONCE per epoch. 'plateau' needs the metric; the deterministic
        # schedules depend only on the epoch count.
        if scheduler is not None:
            if lr_schedule == 'plateau':
                scheduler.step(selection_score)
            else:
                scheduler.step()

        is_best = selection_score > best_selection_score
        if is_best:
            best_selection_score, best_epoch = selection_score, epoch
            best_val_dice = epoch_val_dice
            best_val_macro_dice = epoch_val_macro_dice
            # keep the best weights in RAM so the test set can be scored with them
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if save_checkpoint:
            run_checkpoint_dir.mkdir(parents=True, exist_ok=True)
            state_dict = model.state_dict()
            state_dict['mask_values'] = dataset.mask_values
            torch.save(state_dict, str(run_checkpoint_dir / 'checkpoint_epoch{}.pth'.format(epoch)))
            logging.info(f'Checkpoint {epoch} saved!')
            # keep a separate copy of the best epoch, so you don't have to hunt for it later
            if is_best:
                torch.save(state_dict, str(run_checkpoint_dir / 'best.pth'))

    # 6. Score the held-out test set ONCE, using the best-validation weights.
    # Treat this as a final report, not something to tune against.
    test_dice = None
    per_patient_results = {}

    # Everything below is scored with the best-VALIDATION weights, not the last epoch.
    if best_state is not None:
        model.load_state_dict(best_state)
        logging.info(f'Loaded best weights (epoch {best_epoch}) for final scoring')

    if eval_test and n_test > 0:
        test_loader = DataLoader(test_set, shuffle=False, drop_last=False,
                                 batch_size=batch_size, num_workers=4, pin_memory=True)
        test_dice = float(evaluate(model, test_loader, device, amp))
        logging.info(f'Held-out test Dice = {test_dice:.4f} '
                     f'(val Dice was {best_val_dice:.4f} at epoch {best_epoch})')
        experiment.summary['test_dice'] = test_dice

    # Reportable metrics: per patient, per class, in 3D, Dice + HD95 in mm.
    # The slice-wise numbers above are inflated by empty classes scoring 1.0, so
    # use these for anything you actually publish.
    #
    # val is scored unconditionally -- it is what a loss ablation compares, and
    # gating it on eval_test would leave `eval_test=False` runs with no CSVs at
    # all. test is scored only when eval_test, so an ablation never touches it.
    if per_patient_metrics and hasattr(dataset, 'spacing_for'):
        splits = [('val', val_idx)] if n_val > 0 else []
        if eval_test and n_test > 0:
            splits.append(('test', test_idx))
        for split_name, split_idx in splits:
            rows, summary = metrics.report(
                model, dataset, split_idx, device, n_classes=model.n_classes,
                out_dir=run_checkpoint_dir, split_name=split_name, amp=amp,
                class_names=PHASE_CLASS_NAMES[phase] if class_names is None else class_names,
                batch_size=batch_size,
            )
            per_patient_results[split_name] = {'rows': rows, 'summary': summary}
            for r in summary:
                experiment.summary[f'{split_name}/{r["class_name"]}/dice_mean'] = r['dice_mean']
                experiment.summary[f'{split_name}/{r["class_name"]}/dice_sd'] = r['dice_sd']
                experiment.summary[f'{split_name}/{r["class_name"]}/hd95_mean'] = r['hd95_mm_mean']
                experiment.summary[f'{split_name}/{r["class_name"]}/hd95_sd'] = r['hd95_mm_sd']

            # Best/worst-scoring slice, GT vs. prediction overlaid, for a quick visual
            # sanity check of the best-validation model -- saved next to the checkpoints.
            metrics.save_best_worst_slices(
                model, dataset, split_idx, device, n_classes=model.n_classes,
                out_dir=run_checkpoint_dir, split_name=split_name, amp=amp,
                class_names=PHASE_CLASS_NAMES[phase] if class_names is None else class_names,
                batch_size=batch_size,
            )

    experiment.summary['best_val_dice'] = best_val_dice
    experiment.summary['best_val_macro_dice'] = best_val_macro_dice
    experiment.summary['best_epoch'] = best_epoch
    experiment.finish()

    run_config.update({
        'finished': datetime.datetime.now().isoformat(timespec='seconds'),
        'best_val_dice': best_val_dice,
        'best_val_macro_dice': best_val_macro_dice,
        'best_epoch': best_epoch,
        'final_val_dice': history[-1]['val_dice'] if history else None,
        'test_dice_slicewise': test_dice,
        'history': history,
    })
    with open(run_checkpoint_dir / 'run_config.json', 'w') as f:
        json.dump(run_config, f, indent=2)

    return {
        'run_name': run_name,
        'checkpoint_dir': run_checkpoint_dir,
        'loss': loss_label,
        'lr_schedule': lr_schedule,
        'best_val_dice': best_val_dice,
        'best_val_macro_dice': best_val_macro_dice,
        'best_epoch': best_epoch,
        'final_val_dice': history[-1]['val_dice'] if history else None,
        'test_dice': test_dice,
        'per_patient_metrics': per_patient_results,
        'n_train': n_train,
        'n_val': n_val,
        'n_test': n_test,
        'val_patients': sorted({dataset.index[i][0] for i in val_idx}),
        'test_patients': sorted({dataset.index[i][0] for i in test_idx}),
        'history': history,
    }


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=1, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-5,
                        help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--scale', '-s', type=float, default=0.5, help='Downscaling factor of the images')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=15.0,
                        help='Percent of the patients used as validation (0-100)')
    parser.add_argument('--test', dest='test', type=float, default=15.0,
                        help='Percent of the patients held out as test (0-100)')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--phase', type=str, default='water', choices=['water', 'fat', 'both'],
                        help='Which half of the DICOM volume / which mask labels to train on: '
                             '"water" keeps the 4 chambers (default), "fat" keeps only EAT, '
                             '"both" stacks water+fat as a 2-channel input and predicts all 6 classes '
                             '(4 chambers + EAT)')
    parser.add_argument('--classes', '-c', type=int, default=None,
                        help='Number of classes (output channels, incl. background). '
                             'Defaults to 5 for --phase water, 2 for --phase fat, 6 for --phase both')
    parser.add_argument('--augment', action='store_true', default=False,
                        help='Apply data augmentation to the training split')
    parser.add_argument('--run-name', dest='run_name', type=str, default=None,
                        help='Name for this run (checkpoint subdirectory and wandb run name)')
    parser.add_argument('--split-seed', dest='split_seed', type=int, default=0,
                        help='Seed for the patient shuffle. Keep this fixed across the project, '
                             'or the held-out test set changes between runs')
    parser.add_argument('--no-test-eval', dest='eval_test', action='store_false', default=True,
                        help='Skip scoring the test set after training')
    parser.add_argument('--seed', type=int, default=0,
                        help='Seed for model init, batch order and augmentation. Vary this '
                             '(with --split-seed fixed) to measure run-to-run variance')
    parser.add_argument('--deterministic', action='store_true', default=False,
                        help='Also force deterministic cuDNN algorithms (slower)')
    parser.add_argument('--loss', type=str, default='dice_ce', choices=sorted(LOSS_REGISTRY),
                        help='Loss function / ablation arm (see utils/losses.py). '
                             '"dice_ce" is the corrected per-class batch Dice + CE; '
                             '"dice_ce_legacy" reproduces the pre-fix pooled Dice')
    parser.add_argument('--lr-schedule', dest='lr_schedule', type=str, default='poly',
                        choices=['poly', 'cosine', 'plateau', 'constant'],
                        help='LR schedule, stepped once per epoch. "poly" is nnU-Net\'s '
                             'rule and is deterministic, so every arm of an ablation gets '
                             'an identical LR trajectory')
    parser.add_argument('--select-on', dest='select_on', type=str, default='macro_dice',
                        choices=['macro_dice', 'slice_dice'],
                        help='Metric used to pick best.pth. Keep this identical across '
                             'every arm of an ablation')
    parser.add_argument('--k-folds', dest='k_folds', type=int, default=0,
                        help='If > 0, replace the single fixed validation split with a K-fold '
                             'split of the non-test pool; --fold selects which fold is '
                             'validation (0-indexed). --validation is ignored when this is set. '
                             'The test carve-out is identical to the single-split case at the '
                             'same --split-seed, so an already-frozen test set stays valid. '
                             'Run once per fold (e.g. one process per GPU with --run-name '
                             'and --fold varying) -- this does not loop over folds itself')
    parser.add_argument('--fold', type=int, default=0,
                        help='Which fold (0-indexed) is validation, when --k-folds > 0')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # seed BEFORE building the model -- UNet(...) draws its initial weights from
    # the torch RNG, so seeding afterwards would leave the init random
    set_seed(args.seed, args.deterministic)

    if args.classes is None:
        args.classes = PHASE_N_CLASSES[args.phase]

    # n_channels is 1 for water/fat (single MRI channel), 2 for both (water+fat stacked)
    # n_classes is the number of probabilities you want to get per pixel
    model = UNet(n_channels=PHASE_N_CHANNELS[args.phase], n_classes=args.classes, bilinear=args.bilinear)
    model = model.to(memory_format=torch.channels_last)

    logging.info(f'Network:\n'
                 f'\t{model.n_channels} input channels\n'
                 f'\t{model.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if model.bilinear else "Transposed conv"} upscaling')

    if args.load:
        state_dict = torch.load(args.load, map_location=device)
        state_dict.pop('mask_values', None)
        model.load_state_dict(state_dict)
        logging.info(f'Model loaded from {args.load}')

    model.to(device=device)
    train_kwargs = dict(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        device=device,
        img_scale=args.scale,
        val_percent=args.val / 100,
        test_percent=args.test / 100,
        split_seed=args.split_seed,
        k_folds=args.k_folds,
        fold=args.fold,
        eval_test=args.eval_test,
        amp=args.amp,
        augment=args.augment,
        run_name=args.run_name,
        seed=args.seed,
        deterministic=args.deterministic,
        loss=args.loss,
        select_on=args.select_on,
        lr_schedule=args.lr_schedule,
        phase=args.phase,
    )
    try:
        train_model(model=model, **train_kwargs)
    except torch.cuda.OutOfMemoryError:
        logging.error('Detected OutOfMemoryError! '
                      'Enabling checkpointing to reduce memory usage, but this slows down training. '
                      'Consider enabling AMP (--amp) for fast and memory efficient training')
        torch.cuda.empty_cache()
        model.use_checkpointing()
        train_model(model=model, **train_kwargs)
