"""Segmentation losses, and the registry that names them for the ablation study.

Each entry in LOSS_REGISTRY is one ablation arm, identified by a single string
that gets recorded verbatim in the run's `run_config.json`. Hyperparameters live
in the registry rather than on the command line on purpose: an arm is then fully
reproducible from its name alone, with nothing to reconstruct later about what
alpha or what schedule produced a given set of CSVs. To sweep a hyperparameter,
add another key.

On the corrected Dice
---------------------
`utils.dice_score.dice_loss` (still used by `evaluate.py`, and kept here as the
`*_legacy` arms) flattens batch and class together and reduces to a single
scalar, so it computes ONE Dice over all five classes pooled -- background
included. Background is the overwhelming majority of pixels in a cardiac slice,
so that number is dominated by background agreement and the four chambers barely
reach the gradient.

`DiceLoss` below fixes three things:

  * background is excluded (class 0), leaving cross-entropy to handle it;
  * the reduction is per class, then averaged, so LV does not swamp the atria;
  * numerator and denominator are aggregated over the BATCH before dividing
    ("batch Dice", as nnU-Net does for 2D configs). This matters a lot here:
    33.5% of (slice, class) pairs in this dataset are empty, and with per-slice
    Dice an absent class scores eps/eps ~ 1.0 with essentially no gradient, so a
    false positive on an empty slice costs almost nothing. Pooling over the batch
    puts those false positives in a denominator that contains real ground truth
    from the other slices, where they are actually penalised.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt

from utils.dice_score import dice_loss as _pooled_dice_loss


class SegLoss:
    """Callable loss with an optional per-epoch hook.

    The hook exists for the boundary arm, whose weight is annealed over training.
    Stateless losses simply do not implement it.
    """

    def on_epoch_start(self, epoch: int, total_epochs: int):
        pass

    def __call__(self, logits, target):
        raise NotImplementedError


def _probs_and_onehot(logits, target, n_classes, include_background):
    """Predicted probabilities and one-hot ground truth, both (B, C, H, W) float32.

    Computed in fp32 regardless of autocast: the sums below are the numerically
    fragile part of a Dice-family loss.
    """
    if n_classes == 1:
        probs = torch.sigmoid(logits.float())
        if probs.dim() == 3:
            probs = probs.unsqueeze(1)
        gt = target.float()
        if gt.dim() == 3:
            gt = gt.unsqueeze(1)
        return probs, gt

    probs = F.softmax(logits.float(), dim=1)
    onehot = F.one_hot(target.long(), n_classes).permute(0, 3, 1, 2).float()
    if not include_background:
        probs, onehot = probs[:, 1:], onehot[:, 1:]
    return probs, onehot


def _soft_counts(probs, onehot, batch_dice: bool):
    """Soft true-positive / false-positive / false-negative totals, one per class."""
    dims = (0, 2, 3) if batch_dice else (2, 3)
    tp = (probs * onehot).sum(dims)
    fp = (probs * (1 - onehot)).sum(dims)
    fn = ((1 - probs) * onehot).sum(dims)
    return tp, fp, fn


class CrossEntropy(SegLoss):
    """CE for multi-class, BCE-with-logits for the single-channel binary path."""

    def __init__(self, n_classes, class_weights=None, label_smoothing: float = 0.0):
        self.n_classes = n_classes
        self.class_weights = class_weights
        self.label_smoothing = label_smoothing
        self._fn = None

    def _build(self, device):
        if self.n_classes == 1:
            return nn.BCEWithLogitsLoss()
        w = None
        if self.class_weights is not None:
            w = torch.as_tensor(self.class_weights, dtype=torch.float32, device=device)
        return nn.CrossEntropyLoss(weight=w, label_smoothing=self.label_smoothing)

    def __call__(self, logits, target):
        if self._fn is None:
            self._fn = self._build(logits.device)
        if self.n_classes == 1:
            return self._fn(logits.squeeze(1).float(), target.float())
        return self._fn(logits.float(), target.long())


class DiceLoss(SegLoss):
    """Soft Dice, per class, background excluded, aggregated over the batch.

    See the module docstring for why each of those three choices differs from the
    upstream implementation.
    """

    def __init__(self, n_classes, include_background: bool = False,
                 batch_dice: bool = True, epsilon: float = 1e-5):
        self.n_classes = n_classes
        self.include_background = include_background
        self.batch_dice = batch_dice
        self.epsilon = epsilon

    def __call__(self, logits, target):
        probs, onehot = _probs_and_onehot(logits, target, self.n_classes, self.include_background)
        tp, fp, fn = _soft_counts(probs, onehot, self.batch_dice)
        dice = (2 * tp + self.epsilon) / (2 * tp + fp + fn + self.epsilon)
        return 1 - dice.mean()


class PooledDiceLoss(SegLoss):
    """The pre-fix upstream behaviour, kept so old checkpoints stay reproducible.

    One Dice over batch and all classes pooled, background included. Do not use
    this for new work -- it is here to be the `*_legacy` ablation arm.
    """

    def __init__(self, n_classes):
        self.n_classes = n_classes

    def __call__(self, logits, target):
        if self.n_classes == 1:
            return _pooled_dice_loss(torch.sigmoid(logits.squeeze(1).float()),
                                     target.float(), multiclass=False)
        return _pooled_dice_loss(
            F.softmax(logits.float(), dim=1),
            F.one_hot(target.long(), self.n_classes).permute(0, 3, 1, 2).float(),
            multiclass=True,
        )


class TverskyLoss(SegLoss):
    """Tversky index as a loss; `gamma > 1` makes it Focal Tversky.

    alpha weights false positives, beta weights false negatives, so beta > alpha
    penalises misses harder -- the usual lever for under-segmented thin-walled
    structures (here, most likely the atria). alpha = beta = 0.5 recovers Dice.
    """

    def __init__(self, n_classes, alpha: float = 0.3, beta: float = 0.7,
                 gamma: float = 1.0, include_background: bool = False,
                 batch_dice: bool = True, epsilon: float = 1e-5):
        self.n_classes = n_classes
        self.alpha, self.beta, self.gamma = alpha, beta, gamma
        self.include_background = include_background
        self.batch_dice = batch_dice
        self.epsilon = epsilon

    def __call__(self, logits, target):
        probs, onehot = _probs_and_onehot(logits, target, self.n_classes, self.include_background)
        tp, fp, fn = _soft_counts(probs, onehot, self.batch_dice)
        tversky = (tp + self.epsilon) / (tp + self.alpha * fp + self.beta * fn + self.epsilon)
        loss = 1 - tversky
        if self.gamma != 1.0:
            loss = loss.clamp_min(self.epsilon) ** self.gamma
        return loss.mean()


class BoundaryLoss(SegLoss):
    """Kervadec et al., 'Boundary loss for highly unbalanced segmentation' (2019).

    Integral of the predicted probability against a level-set distance map of the
    ground truth: phi is negative inside the structure and positive outside, so
    probability mass placed far from the target costs in proportion to how far
    away it is. This is the term that targets HD95/ASSD rather than overlap.

    Never use it alone -- it has no region anchor and is unstable on its own.
    Always compound it with Dice/CE, which is what the registry does.

    phi is in pixels, and the distance transforms are computed on CPU per batch
    (~1 ms per class per slice). Classes absent from a slice get phi = 0, which
    makes them contribute nothing rather than an undefined distance.
    """

    def __init__(self, n_classes, include_background: bool = False):
        self.n_classes = n_classes
        self.include_background = include_background

    def _distance_maps(self, target):
        t = target.detach().cpu().numpy().astype(np.int64)
        start = 0 if (self.include_background or self.n_classes == 1) else 1
        n_fg = max(1, self.n_classes - start)
        out = np.zeros((t.shape[0], n_fg) + t.shape[1:], dtype=np.float32)
        for b in range(t.shape[0]):
            for j, c in enumerate(range(start, max(self.n_classes, start + 1))):
                mask = (t[b] == c) if self.n_classes > 1 else (t[b] > 0.5)
                if not mask.any() or mask.all():
                    continue                      # no boundary to measure
                out[b, j] = distance_transform_edt(~mask) - distance_transform_edt(mask)
        return out

    def __call__(self, logits, target):
        probs, _ = _probs_and_onehot(logits, target, self.n_classes, self.include_background)
        phi = torch.from_numpy(self._distance_maps(target)).to(probs.device)
        return (probs * phi).mean()


class Compound(SegLoss):
    """Weighted sum of losses.

    A weight is either a float (constant) or a `(start, end)` pair, which ramps
    linearly across epochs. The ramp is what implements Kervadec's rebalancing:
    the region term decays 1.0 -> 0.01 while the boundary term rises to meet it,
    so training is anchored by Dice/CE early and shaped by the boundary term late.
    """

    def __init__(self, terms):
        self.terms = list(terms)                   # [(loss, weight), ...]
        self._weights = [self._at(w, 0.0) for _, w in self.terms]

    @staticmethod
    def _at(weight, frac):
        if isinstance(weight, (int, float)):
            return float(weight)
        start, end = weight
        return float(start + frac * (end - start))

    def on_epoch_start(self, epoch, total_epochs):
        frac = 0.0 if total_epochs <= 1 else (epoch - 1) / (total_epochs - 1)
        self._weights = []
        for loss, weight in self.terms:
            loss.on_epoch_start(epoch, total_epochs)
            self._weights.append(self._at(weight, frac))

    @property
    def weights(self):
        return list(self._weights)

    def __call__(self, logits, target):
        total = None
        for (loss, _), w in zip(self.terms, self._weights):
            term = w * loss(logits, target)
            total = term if total is None else total + term
        return total


# --- the ablation arms -------------------------------------------------------
#
# Each key is one arm. `dice_ce` is the corrected baseline; `dice_ce_legacy`
# reproduces the pre-fix behaviour so the existing checkpoints remain
# interpretable. Everything is weighted 1:1 against CE except the boundary arm,
# whose schedule is the point of it.

LOSS_REGISTRY = {
    'ce': lambda nc: CrossEntropy(nc),

    'dice': lambda nc: DiceLoss(nc),

    'dice_ce': lambda nc: Compound([(DiceLoss(nc), 1.0),
                                    (CrossEntropy(nc), 1.0)]),

    'dice_ce_legacy': lambda nc: Compound([(PooledDiceLoss(nc), 1.0),
                                           (CrossEntropy(nc), 1.0)]),

    'tversky_ce': lambda nc: Compound([(TverskyLoss(nc, alpha=0.3, beta=0.7), 1.0),
                                       (CrossEntropy(nc), 1.0)]),

    'focal_tversky_ce': lambda nc: Compound([(TverskyLoss(nc, alpha=0.3, beta=0.7, gamma=1.33), 1.0),
                                             (CrossEntropy(nc), 1.0)]),

    'dice_ce_boundary': lambda nc: Compound([(DiceLoss(nc), (1.0, 0.01)),
                                             (CrossEntropy(nc), (1.0, 0.01)),
                                             (BoundaryLoss(nc), (0.0, 0.99))]),
}


def build_loss(spec, n_classes):
    """Resolve a registry key (or an already-built loss object) to a callable."""
    if isinstance(spec, str):
        if spec not in LOSS_REGISTRY:
            raise ValueError(f'Unknown loss {spec!r}. Available: {sorted(LOSS_REGISTRY)}')
        return LOSS_REGISTRY[spec](n_classes)
    if callable(spec):
        return spec
    raise TypeError(f'loss must be a registry key or a callable, got {type(spec).__name__}')


def loss_name(spec):
    """What to record in run_config.json for this loss."""
    return spec if isinstance(spec, str) else type(spec).__name__
