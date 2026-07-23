import argparse
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

import re
from collections import defaultdict
from torch.utils.data import Subset
from torch.utils.data import Sampler
import datetime
import time

dir_img = Path('./data/imgs/')
dir_mask = Path('./data/masks/')
dir_checkpoint = Path('./checkpoints/')


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

def train_model(
        model,
        device,
        epochs: int = 5,
        batch_size: int = 1,
        learning_rate: float = 1e-5,
        val_percent: float = 0.1,
        save_checkpoint: bool = True,
        img_scale: float = 0.5,
        amp: bool = False,
        weight_decay: float = 1e-8,
        momentum: float = 0.9,
        gradient_clipping: float = 1.0,
        run_name=None,
):
    # 1. Create dataset
    #dataset = MRIDataset(dir_img, dir_mask, img_scale)
    #dataset = VolumeMRIDataset(dir_img, dir_mask, scale=img_scale)

    try:
        dataset = VolumeMRIDataset(dir_img, dir_mask, img_scale)
    except (AssertionError, RuntimeError, IndexError):
        print()
        dataset = BasicDataset(dir_img, dir_mask, img_scale)

    # 2. Split into train / validation partitions
    patients = list(dataset.mask_file_for.keys())
    random.Random(0).shuffle(patients)
    n_val_patients = max(1, int(len(patients) * val_percent))
    val_patients = set(patients[:n_val_patients])

    train_idx = [i for i, (p, _) in enumerate(dataset.index) if p not in val_patients]
    val_idx = [i for i, (p, _) in enumerate(dataset.index) if p in val_patients]

    train_set, val_set = Subset(dataset, train_idx), Subset(dataset, val_idx)
    n_train, n_val = len(train_set), len(val_set)

    # 3. Create data loaders
    train_index_subset = [dataset.index[i] for i in train_idx]
    val_index_subset = [dataset.index[i] for i in val_idx]

    train_sampler = PatientGroupedSampler(train_index_subset)
    val_sampler = PatientGroupedSampler(val_index_subset)

    # wrap training set with augmentation
    train_set = AugmentedDataset(train_set)

    loader_args = dict(batch_size=batch_size, num_workers=4, pin_memory=True, persistent_workers=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=False, **loader_args)

    # (Initialize logging)
    experiment = wandb.init(
    project='cardiac-mri-segmentation',
    name='unet-baseline',
    resume='allow'
)
    experiment.config.update(
        dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
             val_percent=val_percent, save_checkpoint=save_checkpoint, img_scale=img_scale, amp=amp)
    )

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
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

                    # loss computed OUTSIDE autocast — always float32
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

        if run_name is None:
            run_name = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        run_checkpoint_dir = Path(dir_checkpoint) / run_name

        if save_checkpoint:
            run_checkpoint_dir.mkdir(parents=True, exist_ok=True)
            state_dict = model.state_dict()
            state_dict['mask_values'] = dataset.mask_values
            torch.save(state_dict, str(run_checkpoint_dir / 'checkpoint_epoch{}.pth'.format(epoch)))
            logging.info(f'Checkpoint {epoch} saved!')


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=1, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-5,
                        help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--scale', '-s', type=float, default=0.5, help='Downscaling factor of the images')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')

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
        del state_dict['mask_values']
        model.load_state_dict(state_dict)
        logging.info(f'Model loaded from {args.load}')

    model.to(device=device)
    try:
        train_model(
            model=model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
            img_scale=args.scale,
            val_percent=args.val / 100,
            amp=args.amp
        )
    except torch.cuda.OutOfMemoryError:
        logging.error('Detected OutOfMemoryError! '
                      'Enabling checkpointing to reduce memory usage, but this slows down training. '
                      'Consider enabling AMP (--amp) for fast and memory efficient training')
        torch.cuda.empty_cache()
        model.use_checkpointing()
        train_model(
            model=model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
            img_scale=args.scale,
            val_percent=args.val / 100,
            amp=args.amp
        )
