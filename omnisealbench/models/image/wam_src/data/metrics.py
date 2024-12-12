# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math

## New for DBSCAN
import numpy as np
import torch
from sklearn.cluster import DBSCAN

from .transforms import image_std


def psnr(x, y):
    """
    Return PSNR
    Args:
        x: Image tensor with values approx. between [-1,1]
        y: Image tensor with values approx. between [-1,1], ex: original image
    """
    delta = x - y
    delta = 255 * (delta * image_std.view(1, 3, 1, 1).to(x.device))
    delta = delta.reshape(-1, x.shape[-3], x.shape[-2], x.shape[-1])  # BxCxHxW
    peak = 20 * math.log10(255.0)
    noise = torch.mean(delta**2, dim=(1, 2, 3))  # B
    psnr = peak - 10 * torch.log10(noise)
    return psnr


def iou(preds, targets, threshold=0.0, label=1):
    """
    Return IoU for a specific label (0 or 1).
    Args:
        preds (torch.Tensor): Predicted masks with shape Bx1xHxW
        targets (torch.Tensor): Target masks with shape Bx1xHxW
        label (int): The label to calculate IoU for (0 for background, 1 for foreground)
        threshold (float): Threshold to convert predictions to binary masks
    """
    preds = preds > threshold  # Bx1xHxW
    targets = targets > 0.5
    if label == 0:
        preds = ~preds
        targets = ~targets
    intersection = (preds & targets).float().sum((1, 2, 3))  # B
    union = (preds | targets).float().sum((1, 2, 3))  # B
    # avoid division by zero
    union[union == 0.0] = intersection[union == 0.0] = 1
    iou = intersection / union
    return iou


def accuracy(
    preds: torch.Tensor, targets: torch.Tensor, threshold: float = 0.0
) -> torch.Tensor:
    """
    Return accuracy
    Args:
        preds (torch.Tensor): Predicted masks with shape Bx1xHxW
        targets (torch.Tensor): Target masks with shape Bx1xHxW
    """
    preds = preds > threshold  # b 1 h w
    targets = targets > 0.5
    correct = (preds == targets).float()  # b 1 h w
    accuracy = torch.mean(correct, dim=(1, 2, 3))  # b
    return accuracy


def bit_accuracy(
    preds: torch.Tensor,
    targets: torch.Tensor,
    masks: torch.Tensor = None,
    threshold: float = 0.0,
) -> torch.Tensor:
    """
    Computes the bit accuracy for each pixel, then averages over all pixels where the mask is not zero.
    This version supports multiple messages and corresponding masks.

    Args:
        preds (torch.Tensor): Predicted bits with shape [B, K, H, W]
        targets (torch.Tensor): Target bits with shape [B, Z, K]
        masks (torch.Tensor): Mask with shape [B, Z, H, W] (optional)
            Used to compute bit accuracy only on non-masked pixels.
            Bit accuracy will be NaN if all pixels are masked.
    """
    if len(targets.shape) != 3:
        print(f"targets.shape: {targets.shape}")
        targets = targets.unsqueeze(1)
    preds = preds > threshold  # B, K, H, W
    targets = targets > 0.5  # B, Z, K
    correct = (
        preds.unsqueeze(1) == targets.unsqueeze(-1).unsqueeze(-1)
    ).float()  # B, Z, K, H, W
    if masks is not None:
        masks = masks.unsqueeze(2)  # B, Z, 1, H, W to align with K dimension
        correct = correct * masks  # Apply masks
        bit_acc = correct.sum() / (masks.sum() * correct.shape[2])
    # Optionally, handle NaNs if all pixels are masked
    # bit_acc = torch.nan_to_num(bit_acc, nan=0.0)
    return bit_acc


def bit_accuracy_dbscan(
    preds: torch.Tensor,
    targets: torch.Tensor,
    masks: torch.Tensor = None,
    threshold: float = 0.0,
) -> torch.Tensor:
    """
    Computes the bit accuracy for each pixel, then averages over all pixels where the mask is not zero.
    This version supports multiple messages and corresponding masks.

    Args:
        preds (torch.Tensor): Predicted bits with shape [B, K, H, W]
        targets (torch.Tensor): Target bits with shape [B, Z, K]
        masks (torch.Tensor): Mask with shape [B, Z, H, W] (optional)
            Used to compute bit accuracy only on non-masked pixels.
            Bit accuracy will be NaN if all pixels are masked.
    """
    if len(targets.shape) != 3:
        print(f"targets.shape: {targets.shape}")
        targets = targets.unsqueeze(1)
    preds = preds > threshold  # B, K, H, W

    union_mask = masks.sum(dim=1)  # B, H, W

    for i in range(preds.shape[0]):
        H, W = union_mask[i].shape
        # select the corresponding mask union and predicition
        mask = union_mask[i]  # H, W
        pred = preds[i]  # K, H, W
        K = pred.shape[0]
        pred = pred.view(K, -1).t().cpu()  # H*W, K

        # pred = pred.detach().float().cpu().numpy() # K, H, W
        # pred = np.transpose(pred, (1, 2, 0)).reshape(-1, 1) # H*W, K
        # valid_indices = mask.view(-1).cpu().numpy() > 0  # shape [H*W], Indices of unmasked data
        valid_indices = (mask.view(-1) > 0).cpu()
        valid_pred = pred[valid_indices].float()  # shape [num_valid, K]
        # scaler = StandardScaler()
        # valid_pred = scaler.fit_transform(valid_pred)
        db = DBSCAN(eps=0.5, min_samples=100).fit(valid_pred)
        labels = torch.Tensor(db.labels_).float()
        # Store labels in the original shape, respecting the mask
        full_labels = -torch.ones(
            pred.shape[0]
        ).float()  # shape [H*W], Start with all as noise
        full_labels[valid_indices] = labels  # Only fill where mask was 1
        full_labels = full_labels.reshape(H, W)
        batch_labels.append(full_labels)

        unique_labels = np.unique(labels)
        unique_labels = unique_labels[unique_labels != -1]  # Exclude noise
        for label in unique_labels:
            cluster_points = valid_pred[labels == label]
            centroid = cluster_points.mean(0)
            # unscale the centroid
            centroid = centroid > 0.5
            representants.append(centroid)
            positions = (full_labels == label).float()
            better_match = 0
            better_match_message = 0
            for i in range(masks.shape[1]):
                intersection = torch.logical_and(positions, masks[i].cpu()).sum().item()
                union = torch.logical_or(positions, masks[i].cpu()).sum().item()
                jaccard_similarity = intersection / union
                if jaccard_similarity > better_match:
                    better_match_message = i
                    better_match = jaccard_similarity
            bit_accuracy = (
                (centroid == targets[i][better_match_message].cpu())
                .float()
                .mean()
                .item()
            )

    targets = targets > 0.5  # B, Z, K
    correct = (
        preds.unsqueeze(1) == targets.unsqueeze(-1).unsqueeze(-1)
    ).float()  # B, Z, K, H, W
    if masks is not None:
        masks = masks.unsqueeze(2)  # B, Z, 1, H, W to align with K dimension
        correct = correct * masks  # Apply masks
        bit_acc = correct.sum() / (masks.sum() * correct.shape[2])
    # Optionally, handle NaNs if all pixels are masked
    # bit_acc = torch.nan_to_num(bit_acc, nan=0.0)
    return bit_acc


def bit_accuracy_1msg(
    preds: torch.Tensor,
    targets: torch.Tensor,
    masks: torch.Tensor = None,
    threshold: float = 0.0,
) -> torch.Tensor:
    """
    Computes the bit accuracy for each pixel, then averages over all pixels.
    Better for "k-bit" evaluation during training since it's independent of detection performance.
    Args:
        preds (torch.Tensor): Predicted bits with shape BxKxHxW
        targets (torch.Tensor): Target bits with shape BxK
        masks (torch.Tensor): Mask with shape Bx1xHxW (optional)
            Used to compute bit accuracy only on non masked pixels.
            Bit accuracy will be NaN if all pixels are masked.
    """
    preds = preds > threshold  # b k h w
    targets = targets > 0.5  # b k
    correct = (preds == targets.unsqueeze(-1).unsqueeze(-1)).float()  # b k h w
    if masks is not None:
        bsz, nbits, h, w = preds.size()
        masks = masks.expand_as(correct).bool()
        correct_list = [correct[i].masked_select(masks[i]) for i in range(len(masks))]
        bit_acc = torch.tensor(
            [torch.mean(correct_list[i]).item() for i in range(len(correct_list))]
        )
    else:
        bit_acc = torch.mean(correct, dim=(1, 2, 3))  # b
    return bit_acc


def bit_accuracy_inference(
    preds: torch.Tensor,
    targets: torch.Tensor,
    masks: torch.Tensor,
    method: str = "hard",
    nb_repetitions: int = 1,
    threshold: float = 0.0,
) -> torch.Tensor:
    """
    Computes the message by averaging over all pixels, then computes the bit accuracy.
    Closer to how the model is evaluated during inference.
    Args:
        preds (torch.Tensor): Predicted bits with shape BxKxHxW
        targets (torch.Tensor): Target bits with shape BxK
        masks (torch.Tensor): Mask with shape Bx1xHxW
            Used to compute bit accuracy only on non masked pixels.
            Bit accuracy will be NaN if all pixels are masked.
        method (str): Method to compute bit accuracy. Options: 'hard', 'soft'
    """
    assert preds.shape[1] % nb_repetitions == 0, preds.shape[1] % nb_repetitions
    a = preds.shape[1] // nb_repetitions
    for i in range(nb_repetitions - 1):
        preds[:, :a, :, :] += preds[:, (1 + i) * a : (i + 2) * a, :, :]
    preds = preds[:, :a, :, :]
    targets = targets[:, :a]  # b k//nb_repetitions

    if method == "hard":
        # convert every pixel prediction to binary, select based on masks, and average
        preds = preds > threshold  # b k h w
        bsz, nbits, h, w = preds.size()
        masks = masks > 0.5  # b 1 h w
        masks = masks.expand_as(preds).bool()
        # masked select only works if all masks in the batch share the same number of 1s
        # not the case here, so we need to loop over the batch
        preds = [
            pred.masked_select(mask).view(nbits, -1) for mask, pred in zip(masks, preds)
        ]  # b k n
        preds = [pred.mean(dim=-1, dtype=float) for pred in preds]  # b k
        preds = torch.stack(preds, dim=0)  # b k
    elif method == "semihard":
        # select every pixel prediction based on masks, and average
        bsz, nbits, h, w = preds.size()
        masks = masks > 0.5  # b 1 h w
        masks = masks.expand_as(preds).bool()
        # masked select only works if all masks in the batch share the same number of 1s
        # not the case here, so we need to loop over the batch
        preds = [
            pred.masked_select(mask).view(nbits, -1) for mask, pred in zip(masks, preds)
        ]  # b k n
        preds = [pred.mean(dim=-1, dtype=float) for pred in preds]  # b k
        preds = torch.stack(preds, dim=0)  # b k
    elif method == "soft":
        # average every pixel prediction, use masks "softly" as weights for averaging
        bsz, nbits, h, w = preds.size()
        masks = masks.expand_as(preds)  # b k h w
        preds = torch.sum(preds * masks, dim=(2, 3)) / torch.sum(
            masks, dim=(2, 3)
        )  # b k
    preds = preds > threshold  # b k
    targets = targets > 0.5  # b k
    correct = (preds == targets).float()  # b k
    bit_acc = torch.mean(correct, dim=(1))  # b
    return bit_acc


def msg_predict_inference(
    preds: torch.Tensor,
    masks: torch.Tensor,
    method: str = "hard",
    threshold: float = 0.0,
) -> torch.Tensor:
    """
    Computes the message by averaging over all pixels, then computes the bit accuracy.
    Closer to how the model is evaluated during inference.
    Args:
        preds (torch.Tensor): Predicted bits with shape BxKxHxW
        targets (torch.Tensor): Target bits with shape BxK
        masks (torch.Tensor): Mask with shape Bx1xHxW
            Used to compute bit accuracy only on non masked pixels.
            Bit accuracy will be NaN if all pixels are masked.
        method (str): Method to compute bit accuracy. Options: 'hard', 'soft'
    """
    if method == "hard":
        # convert every pixel prediction to binary, select based on masks, and average
        preds = preds > threshold  # b k h w
        bsz, nbits, h, w = preds.size()
        masks = masks > 0.5  # b 1 h w
        masks = masks.expand_as(preds).bool()
        # masked select only works if all masks in the batch share the same number of 1s
        # not the case here, so we need to loop over the batch
        preds = [
            pred.masked_select(mask).view(nbits, -1) for mask, pred in zip(masks, preds)
        ]  # b k n
        preds = [pred.mean(dim=-1, dtype=float) for pred in preds]  # b k
        preds = torch.stack(preds, dim=0)  # b k
    elif method == "semihard":
        # select every pixel prediction based on masks, and average
        bsz, nbits, h, w = preds.size()
        masks = masks > 0.5  # b 1 h w
        masks = masks.expand_as(preds).bool()
        # masked select only works if all masks in the batch share the same number of 1s
        # not the case here, so we need to loop over the batch
        preds = [
            pred.masked_select(mask).view(nbits, -1) for mask, pred in zip(masks, preds)
        ]  # b k n
        preds = [pred.mean(dim=-1, dtype=float) for pred in preds]  # b k
        preds = torch.stack(preds, dim=0)  # b k
    elif method == "soft":
        # average every pixel prediction, use masks "softly" as weights for averaging
        bsz, nbits, h, w = preds.size()
        masks = masks.expand_as(preds)  # b k h w
        preds = torch.sum(preds * masks, dim=(2, 3)) / torch.sum(
            masks, dim=(2, 3)
        )  # b k
    preds = preds > 0.5  # b k
    return preds


def bit_accuracy_mv(
    preds: torch.Tensor,
    targets: torch.Tensor,
    masks: torch.Tensor = None,
    threshold: float = 0.0,
) -> torch.Tensor:
    """
    (Majority vote)
    Return bit accuracy
    Args:
        preds (torch.Tensor): Predicted bits with shape BxKxHxW
        targets (torch.Tensor): Target bits with shape BxK
        masks (torch.Tensor): Mask with shape Bx1xHxW (optional)
            Used to compute bit accuracy only on non masked pixels.
            Bit accuracy will be NaN if all pixels are masked.
    """
    preds = preds > threshold  # b k h w
    targets = targets > 0.5  # b k
    correct = (preds == targets.unsqueeze(-1).unsqueeze(-1)).float()  # b k h w
    if masks is not None:
        bsz, nbits, h, w = preds.size()
        masks = masks.expand_as(correct).bool()
        preds = preds.masked_select(masks).view(bsz, nbits, -1)  # b k n
        # correct = correct.masked_select(masks).view(bsz, nbits, -1)  # b k n
        # correct = correct.unsqueeze(-1)  # b k n 1
    # Perform majority vote for each bit
    preds_majority, _ = torch.mode(preds, dim=-1)  # b k
    # Compute bit accuracy
    correct = (preds_majority == targets).float()  # b k
    # bit_acc = torch.mean(correct, dim=(1,2,3))  # b
    bit_acc = torch.mean(correct, dim=-1)  # b
    return bit_acc


def bit_accuracy_repeated(
    preds: torch.Tensor,
    targets: torch.Tensor,
    nb_repetitions,
    masks: torch.Tensor = None,
    threshold: float = 0.0,
    factor: int = 2,
) -> torch.Tensor:
    """
    Return bit accuracy
    Args:
        preds (torch.Tensor): Predicted bits with shape BxKxHxW
        targets (torch.Tensor): Target bits with shape BxK
        masks (torch.Tensor): Mask with shape Bx1xHxW (optional)
            Used to compute bit accuracy only on non masked pixels.
            Bit accuracy will be NaN if all pixels are masked.
    """

    assert preds.shape[1] % nb_repetitions == 0, preds.shape[1] % nb_repetitions
    a = preds.shape[1] // nb_repetitions
    for i in range(nb_repetitions - 1):
        preds[:, :a, :, :] += preds[:, (1 + i) * a : (i + 2) * a, :, :]
    preds = (preds > threshold)[:, :a, :, :]
    targets = (targets > 0.5)[:, :a]  # b k//nb_repetitions
    correct = (preds == targets.unsqueeze(-1).unsqueeze(-1)).float()  # b k' h w
    if masks is not None:
        masks = masks[:, :a, :, :]
        bsz, nbits, h, w = preds.size()
        masks = masks.expand_as(correct).bool()
        preds = preds.masked_select(masks).view(bsz, nbits, -1)  # b k' n
        correct = correct.masked_select(masks).view(bsz, nbits, -1)  # b k' n
        correct = correct.unsqueeze(-1)  # b k' n 1
    # Compute bit accuracy
    bit_acc = torch.mean(correct, dim=(1, 2, 3))  # b
    return bit_acc
