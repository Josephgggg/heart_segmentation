""" MONAI's Attention U-Net (Oktay et al. 2018), wrapped to this repo's model interface.

This is a WHOLE-NETWORK swap, not attention gates bolted onto `unet_model.UNet`.
`monai.networks.nets.AttentionUnet` brings its own encoder/decoder, and it differs
from `UNet` in three ways that all affect the result:

  * downsampling is a stride-2 convolution, not `nn.MaxPool2d` (`Down`);
  * upsampling is always a transposed convolution -- MONAI has no bilinear
    variant, so `bilinear=True` is rejected rather than silently ignored;
  * each skip connection passes through an attention gate, and gated skip and
    upsampled decoder feature are then fused by a 3x3 conv rather than
    concatenated straight into a `DoubleConv`.

The conv blocks themselves line up with `DoubleConv` (two convs, BatchNorm,
ReLU; MONAI's `adn_ordering="NDA"` puts norm before activation the same way),
and `DEFAULT_CHANNELS` reproduces UNet's 64/128/256/512/1024 widths at the same
depth, and the parameter counts come out close (31.74 M vs UNet's 31.04 M at
n_channels=2, n_classes=6 -- the gates and merge convs are 1x1/3x3 and cheap).
Even so, an attention_unet-vs-unet comparison is NOT a clean attention ablation:
the blocks, the downsampling and the skip fusion all changed at once. Read a
difference as "this architecture vs that architecture", not as "attention helps".

Input size: MONAI concatenates skip and upsampled features with no padding or
cropping, so every spatial dimension has to survive `len(channels) - 1` halvings
and be doubled back exactly -- i.e. be a multiple of 2 ** (len(channels) - 1),
16 for the default 5 levels. At the project default `--scale 0.5` the slices are
432x432 (432 = 16 * 27), so nothing happens; `forward` zero-pads the bottom/right
up to that multiple and crops the logits back, so that a different `--scale` is
not a shape crash from inside MONAI. `UNet` gets this for free via `Up`'s F.pad.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

from monai.networks.nets import AttentionUnet as _MonaiAttentionUnet

# Encoder widths, top block first -- deliberately UNet's, so depth and width are
# not a second thing that differs between the two architectures.
DEFAULT_CHANNELS = (64, 128, 256, 512, 1024)


def _checkpointed(module):
    """Return a forward that recomputes `module` in the backward pass.

    Bound onto the INSTANCE as `module.forward`, which shadows the class method
    without wrapping the module in a new one -- wrapping would insert an extra
    level into every parameter name, and `train.py` saves raw `state_dict()`
    keys, so a checkpointed run's best.pth would no longer load into a normally
    built model.
    """
    inner = module.forward

    def forward(*args):
        if module.training and torch.is_grad_enabled():
            return torch.utils.checkpoint.checkpoint(inner, *args, use_reentrant=False)
        return inner(*args)

    return forward


class AttentionUNet(nn.Module):
    """Drop-in replacement for `UNet` exposing `n_channels` / `n_classes` / `bilinear`.

    Those three attributes are what the rest of the codebase reads off a model
    (`train.py`'s run_config and channel assertion, `evaluate.py`, `predict.py`),
    so anything else that takes a model keeps working unchanged.
    """

    def __init__(self, n_channels, n_classes, bilinear=False,
                 channels=DEFAULT_CHANNELS, dropout=0.0):
        super().__init__()
        if bilinear:
            raise ValueError(
                'AttentionUNet has no bilinear-upsampling variant -- MONAI always uses '
                'transposed convolutions. Drop --bilinear, or use --arch unet.')
        channels = tuple(channels)
        if len(channels) < 2:
            raise ValueError(f'channels needs at least 2 levels, got {channels}')

        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = False
        self.arch = 'attention_unet'          # recorded in run_config.json
        self.channels = channels
        self.strides = (2,) * (len(channels) - 1)
        # Every spatial dim must be divisible by this for the skip concatenations
        # to line up -- see the module docstring.
        self.size_multiple = 2 ** len(self.strides)

        self.net = _MonaiAttentionUnet(
            spatial_dims=2,
            in_channels=n_channels,
            out_channels=n_classes,
            channels=channels,
            strides=self.strides,
            dropout=dropout,
        )

    def forward(self, x):
        h, w = x.shape[-2:]
        m = self.size_multiple
        pad_h, pad_w = (-h) % m, (-w) % m
        if pad_h or pad_w:
            # bottom/right only, so the crop below is a plain slice
            x = F.pad(x, (0, pad_w, 0, pad_h))

        logits = self.net(x)

        if pad_h or pad_w:
            logits = logits[..., :h, :w]
        return logits

    def use_checkpointing(self):
        """Trade compute for memory, for `train.py`'s OutOfMemoryError retry.

        Checkpoints each `ConvBlock`/`UpConv` separately. Checkpointing the
        network as a single segment -- the obvious way to do it, given MONAI
        builds the decoder as one recursive `AttentionLayer` chain -- saves
        nothing: the whole forward is recomputed at once during backward, so
        peak memory is unchanged and it only costs an extra forward pass.
        Measured at batch 4, 432x432: 5.57 GiB plain, 5.57 GiB checkpointed as
        one segment, 4.19 GiB per block. Returns the number of blocks wrapped.
        """
        from monai.networks.nets.attentionunet import ConvBlock, UpConv

        n = 0
        for m in self.net.modules():
            if isinstance(m, (ConvBlock, UpConv)):
                m.forward = _checkpointed(m)
                n += 1
        return n
