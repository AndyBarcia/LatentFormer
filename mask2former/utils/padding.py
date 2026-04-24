# Copyright (c) Facebook, Inc. and its affiliates.
import torch
from torch.nn import functional as F


def image_padding_mask(images):
    batch_size, _, padded_h, padded_w = images.tensor.shape
    mask = torch.zeros(
        (batch_size, padded_h, padded_w),
        dtype=torch.bool,
        device=images.tensor.device,
    )
    for idx, (image_h, image_w) in enumerate(images.image_sizes):
        valid_h = min(image_h, padded_h)
        valid_w = min(image_w, padded_w)
        mask[idx, valid_h:, :] = True
        mask[idx, :, valid_w:] = True
    return mask


def resize_padding_mask(mask, size):
    if mask is None:
        return None
    if isinstance(mask, (list, tuple)):
        for candidate in mask:
            if candidate.shape[-2:] == size:
                return candidate
        mask = mask[0]
    if mask.shape[-2:] == size:
        return mask
    return F.interpolate(mask[:, None].float(), size=size, mode="nearest")[:, 0].to(torch.bool)


def nonempty_padding_mask(mask):
    return mask if mask is not None and mask.any() else None
