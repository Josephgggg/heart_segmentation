""" The milesial UNet with MONAI's attention gate on each skip connection.

This is the clean single-variable ablation against `unet_model.UNet`: same
`DoubleConv`/`Down`/`OutConv` blocks, same 64/128/256/512/1024 widths, same
maxpool downsampling, same `bilinear` handling, same `Up` padding rule. The
ONLY change is that each skip connection passes through
`monai.networks.nets.attentionunet.AttentionBlock` -- Oktay et al. 2018's
attention gate, MONAI's implementation of it -- before being concatenated.

Contrast with `attention_unet.py`, which swaps in MONAI's whole `AttentionUnet`
network and therefore changes the blocks, the downsampling and the skip fusion
all at once. Use THIS one to ask "do attention gates help"; use that one to ask
"how does the standard reference implementation compare".

Gate wiring follows the paper: the gating signal g is the coarser decoder
feature (after upsampling), and the gate reweights the encoder skip x. In both
`bilinear` modes `Up.up` leaves the decoder feature with `in_channels // 2`
channels, which is exactly the skip's channel count, so f_g == f_l and the
intermediate width f_int is half that -- the same ratio MONAI's own
`AttentionLayer` uses.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from monai.networks.nets.attentionunet import AttentionBlock

from .unet_parts import DoubleConv, Down, Up, OutConv


class GatedUp(Up):
    """`Up`, with the skip connection gated before concatenation."""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__(in_channels, out_channels, bilinear)
        # channels of the upsampled decoder feature, and of the skip -- equal in
        # both bilinear and transposed-conv modes (see module docstring)
        c = in_channels // 2
        self.gate = AttentionBlock(spatial_dims=2, f_g=c, f_l=c, f_int=max(c // 2, 1))

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # identical padding to Up.forward -- the gate needs the two feature maps
        # to be the same size, so this has to happen before gating, not after
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x2 = self.gate(g=x1, x=x2)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class AttentionGateUNet(nn.Module):
    """UNet with attention-gated skips. Mirrors `UNet` exactly apart from `GatedUp`."""

    def __init__(self, n_channels, n_classes, bilinear=False):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.arch = 'unet_attn_gates'          # recorded in run_config.json

        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))
        self.up1 = (GatedUp(1024, 512 // factor, bilinear))
        self.up2 = (GatedUp(512, 256 // factor, bilinear))
        self.up3 = (GatedUp(256, 128 // factor, bilinear))
        self.up4 = (GatedUp(128, 64, bilinear))
        self.outc = (OutConv(64, n_classes))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

    def use_checkpointing(self):
        """Trade compute for memory, for `train.py`'s OutOfMemoryError retry.

        Checkpoints each encoder/decoder stage separately -- unlike `UNet`'s own
        `use_checkpointing()`, which calls `torch.utils.checkpoint(module)` and
        would raise (that is a module, not the `checkpoint()` function). Bound
        onto the instances so `state_dict()` keys are unchanged and a
        checkpointed run's best.pth still loads into a normally built model.
        """
        import torch.utils.checkpoint as cp

        n = 0
        for m in (self.inc, self.down1, self.down2, self.down3, self.down4,
                  self.up1, self.up2, self.up3, self.up4):
            inner = m.forward

            def fwd(*args, _inner=inner, _m=m):
                if _m.training and torch.is_grad_enabled():
                    return cp.checkpoint(_inner, *args, use_reentrant=False)
                return _inner(*args)

            m.forward = fwd
            n += 1
        return n
