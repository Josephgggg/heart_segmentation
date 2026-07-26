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
from utils.dice_score import dice_loss
from utils import metrics

import re
from collections import defaultdict
from torch.utils.data import Subset
from torch.utils.data import Sampler
import datetime
import time

dir_img = Path('./data/imgs/')
dir_mask = Path('./data/masks/')
dir_checkpoint = Path('./checkpoints/')

# Cardiac chamber labels in the NIfTI masks. 0 = background; label 5 (fat) is
# relabelled to background by VolumeMRIDataset.
CLASS_NAMES = {1: 'LV', 2: 'RV', 3: 'LA', 4: 'RA'}


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


def build_dataset(img_scale: float = 0.5):
    try:
        return VolumeMRIDataset(dir_img, dir_mask, img_scale)
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
):
    # 1. Create dataset
    if dataset is None:
        dataset = build_dataset(img_scale)

    # 2. Split into train / validation / test partitions, by patient
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

    loader_args = dict(batch_size=batch_size, num_workers=4, pin_memory=True, persistent_workers=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=False, **loader_args)
    # the test loader is built once at the very end, so it gets no persistent workers

    if run_name is None:
        run_name = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    run_checkpoint_dir = Path(dir_checkpoint) / run_name

    # Record exactly what produced this run, next to its metrics. Without this,
    # a directory of CSVs from a dozen experiments is unattributable later.
    run_config = {
        'run_name': run_name,
        'started': datetime.datetime.now().isoformat(timespec='seconds'),
        'epochs': epochs, 'batch_size': batch_size, 'learning_rate': learning_rate,
        'weight_decay': weight_decay, 'momentum': momentum,
        'gradient_clipping': gradient_clipping,
        'optimizer': 'RMSprop', 'scheduler': 'ReduceLROnPlateau(max, patience=5)',
        'img_scale': img_scale, 'amp': amp, 'augment': augment,
        'n_classes': model.n_classes, 'n_channels': model.n_channels,
        'bilinear': model.bilinear,
        'val_percent': val_percent, 'test_percent': test_percent, 'split_seed': split_seed,
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
        reinit=True,
    )
    experiment.config.update(
        dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
             val_percent=val_percent, test_percent=test_percent, split_seed=split_seed,
             save_checkpoint=save_checkpoint, img_scale=img_scale, amp=amp, augment=augment,
             n_train=n_train, n_val=n_val, n_test=n_test)
    )

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Test size:       {n_test} (held out, {'scored once after training' if eval_test else 'not scored'})
        Augmentation:    {augment}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Images scaling:  {img_scale}
        Mixed Precision: {amp}
    ''')

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    optimizer = optim.RMSprop(model.parameters(),
                              lr=learning_rate, weight_decay=weight_decay, momentum=momentum, foreach=True)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)  # goal: maximize Dice score
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    criterion = nn.CrossEntropyLoss() if model.n_classes > 1 else nn.BCEWithLogitsLoss()
    global_step = 0
    batch_skip_count = 0
    best_val_dice = 0.0
    best_epoch = 0
    best_state = None
    history = []

    # 5. Begin training
    for epoch in range(1, epochs + 1):
        model.train()
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
                    #print("hello")
                    masks_pred = model(images)
                    #print("bye")
                #masks_pred = torch.clamp(masks_pred.float(), min=-20, max=20)

                    if model.n_classes == 1:
                        loss = criterion(masks_pred.squeeze(1).float(), true_masks.float())
                        loss += dice_loss(F.sigmoid(masks_pred.squeeze(1).float()), true_masks.float(), multiclass=False)
                    else:
                        loss = criterion(masks_pred.float(), true_masks)
                        loss += dice_loss(
                            F.softmax(masks_pred.float(), dim=1),
                            F.one_hot(true_masks, model.n_classes).permute(0, 3, 1, 2).float(),
                            multiclass=True
                        )

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

                        val_score = evaluate(model, val_loader, device, amp)
                        scheduler.step(val_score)


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

        # NEW — compute Dice once, cleanly, at the end of this epoch
        epoch_val_dice = evaluate(model, val_loader, device, amp)
        mean_epoch_loss = epoch_loss / len(train_loader)

        logging.info(f'Epoch {epoch}: mean train loss = {mean_epoch_loss:.4f}, val Dice = {epoch_val_dice:.4f}')

        epoch_val_dice = float(epoch_val_dice)
        history.append({'epoch': epoch, 'train_loss': mean_epoch_loss, 'val_dice': epoch_val_dice})

        experiment.log({
            'epoch': epoch,
            'epoch_train_loss': mean_epoch_loss,
            'epoch_val_dice': epoch_val_dice,
        })

        is_best = epoch_val_dice > best_val_dice
        if is_best:
            best_val_dice, best_epoch = epoch_val_dice, epoch
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
    if eval_test and n_test > 0:
        if best_state is not None:
            model.load_state_dict(best_state)
            logging.info(f'Loaded best weights (epoch {best_epoch}) for test evaluation')
        test_loader = DataLoader(test_set, shuffle=False, drop_last=False,
                                 batch_size=batch_size, num_workers=4, pin_memory=True)
        test_dice = float(evaluate(model, test_loader, device, amp))
        logging.info(f'Held-out test Dice = {test_dice:.4f} '
                     f'(val Dice was {best_val_dice:.4f} at epoch {best_epoch})')
        experiment.summary['test_dice'] = test_dice

        # Reportable metrics: per patient, per class, in 3D, Dice + HD95 in mm.
        # The slice-wise number above is inflated by empty classes scoring 1.0,
        # so use these for anything you actually publish.
        if per_patient_metrics and hasattr(dataset, 'spacing_for'):
            for split_name, split_idx in (('val', val_idx), ('test', test_idx)):
                rows, summary = metrics.report(
                    model, dataset, split_idx, device, n_classes=model.n_classes,
                    out_dir=run_checkpoint_dir, split_name=split_name, amp=amp,
                    class_names=CLASS_NAMES if class_names is None else class_names,
                    batch_size=batch_size,
                )
                per_patient_results[split_name] = {'rows': rows, 'summary': summary}
                for r in summary:
                    experiment.summary[f'{split_name}/{r["class_name"]}/dice_mean'] = r['dice_mean']
                    experiment.summary[f'{split_name}/{r["class_name"]}/dice_sd'] = r['dice_sd']
                    experiment.summary[f'{split_name}/{r["class_name"]}/hd95_mean'] = r['hd95_mm_mean']
                    experiment.summary[f'{split_name}/{r["class_name"]}/hd95_sd'] = r['hd95_mm_sd']

    experiment.summary['best_val_dice'] = best_val_dice
    experiment.summary['best_epoch'] = best_epoch
    experiment.finish()

    run_config.update({
        'finished': datetime.datetime.now().isoformat(timespec='seconds'),
        'best_val_dice': best_val_dice,
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
        'best_val_dice': best_val_dice,
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
    parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')
    parser.add_argument('--augment', action='store_true', default=False,
                        help='Apply data augmentation to the training split')
    parser.add_argument('--run-name', dest='run_name', type=str, default=None,
                        help='Name for this run (checkpoint subdirectory and wandb run name)')
    parser.add_argument('--split-seed', dest='split_seed', type=int, default=0,
                        help='Seed for the patient shuffle. Keep this fixed across the project, '
                             'or the held-out test set changes between runs')
    parser.add_argument('--no-test-eval', dest='eval_test', action='store_false', default=True,
                        help='Skip scoring the test set after training')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    model = UNet(n_channels=1, n_classes=args.classes, bilinear=args.bilinear)
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
        eval_test=args.eval_test,
        amp=args.amp,
        augment=args.augment,
        run_name=args.run_name,
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
