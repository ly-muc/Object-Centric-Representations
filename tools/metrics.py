import torch

from scipy.optimize import linear_sum_assignment
from torch import Tensor
from typing import Optional


def adjusted_rand_index(
    predicted_oh: Tensor,
    ground_truth_oh: Tensor,
    n_points: Optional[int] = None,
    mean: bool = False
):
    """Implements adjusted rand index metric. 
    Description: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.adjusted_rand_score.html

    Args:
        predicted_oh (Tensor): Predicted mask
        ground_truth_oh (Tensor): Ground truth mask
        n_points (int): Number of prediction points
        mean (bool, optional): Return mean value. Defaults to False.

    Returns:
        Tensor: ari score
    """
    assert predicted_oh.shape == ground_truth_oh.shape, \
        "Inputs must have same shape!"

    # Check datatype
    if predicted_oh.dtype is not torch.float:
        predicted_oh = predicted_oh.float()

    if ground_truth_oh.dtype is not torch.float:
        ground_truth_oh = ground_truth_oh.float()

    if n_points is None:
        n_points = predicted_oh.shape[1]

    # compute contingency table 
    contingency = torch.einsum('bji,bjk->bki', predicted_oh, ground_truth_oh)

    a = torch.sum(contingency, 1)
    b = torch.sum(contingency, 2)

    rindex = torch.sum(contingency * (contingency - 1), dim=[1, 2])
    aindex = torch.sum(a * (a - 1), dim=1)  
    bindex = torch.sum(b * (b - 1), dim=1)

    expected_rindex = aindex * bindex / (n_points*(n_points-1))
    max_rindex = (aindex + bindex) / 2
    ari = (rindex - expected_rindex) / (max_rindex - expected_rindex)

    if mean:
        ari = torch.mean(ari)

    return ari


def fg_adjusted_rand_index(
    predicted_oh: Tensor,
    ground_truth_oh: Tensor,
    foreground: Tensor,
    n_points: int,
    mean: bool = False
):

    # predicted_oh[:, ~foreground] = 0.
    # ground_truth_oh [:, ~foreground] = 0.

    # n_points -= torch.sum(foreground)

    ari = adjusted_rand_index(predicted_oh, ground_truth_oh, n_points, mean)

    return ari


def mean_iou(
    predicted_oh: Tensor,
    ground_truth_oh: Tensor,
    mean: bool = False
):
    """Computes mean IoU with respect to the number of objects in scene.

    Args:
        predicted_oh (Tensor): Predicted mask
        ground_truth_oh (Tensor): Ground truth mask
        mean (bool): Return mean value. Defaults to False.
    Returns:
        Tensor: mean iou
    """
    # assert predicted_oh.shape == ground_truth_oh.shape, \
    # "Inputs must have same shape!"

    # Check datatype
    if predicted_oh.dtype is not torch.float:
        predicted_oh = predicted_oh.float()

    if ground_truth_oh.dtype is not torch.float:
        ground_truth_oh = ground_truth_oh.float()
    
    B, _, M = ground_truth_oh.shape

    iou_vals = torch.zeros(B, M, M, device=predicted_oh.device)
    
    intersection = torch.einsum('bjk,bji->bki', predicted_oh, ground_truth_oh)

    # non zero entries
    nz_mask = torch.sum(ground_truth_oh, dim=1) > 0

    # iterate over all ground truth segmentation
    for s in range(M):
        gt = ground_truth_oh[..., s]

        # shape: [batch, slots]
        union = torch.sum((gt.unsqueeze(-1) + predicted_oh) > 0, dim=1)

        # ignore zero values for union
        iou_vals[nz_mask[:, s], :, s] = intersection[nz_mask[:, s], :, s] \
            / union[nz_mask[:, s]]

    # number of objects per image
    num_objects = torch.sum(nz_mask, dim=1).cpu()

    iou_vals = iou_vals.cpu()
    
    results = []

    for n_obj, vals in zip(num_objects, iou_vals):

        row, col = linear_sum_assignment(1 - vals)

        results.append(torch.sum(vals[row, col]) / n_obj)

    m_iou = torch.Tensor(results)

    if mean:
        m_iou = torch.mean(m_iou)

    return m_iou


def get_onehot(
    segmentation: Tensor,
    mask: Tensor,
    batch_size: int,
    max_num_objects: int
):
    """Function returns one hot encoded segmentation and mask
    Args:
        segmenation (Tensor): Ground Truth. Shape:
            [batch_size, 1, height, width]
        mask (Tensor): Predicted mask. Shape:
            [batch_size, num_slots, height, width]
    Returns:
        tuple: One hot encoded segmentation and mask
    """
    slot_attns = torch.argmax(mask, axis=1)
    seg_oh = torch.nn.functional.one_hot(segmentation.view(batch_size,
            -1).to(torch.int64), max_num_objects)
    slot_oh = torch.nn.functional.one_hot(slot_attns.view(batch_size,
            -1).to(torch.int64), max_num_objects)

    return (seg_oh, slot_oh)


def get_masks_onehot(mask: Tensor, batch_size: int, max_num_objects: int):
    
    """Function returns one hot encoded masks
    Args:
        mask (Tensor): Predicted mask. Shape:
            [batch_size, num_slots, height, width]
    Returns:
        tuple: One hot encoded masks
    """
    
    slot_attns = torch.argmax(mask, axis=1)
    slot_oh = torch.nn.functional.one_hot(slot_attns.view(batch_size,
            -1).to(torch.int64), max_num_objects)

    return slot_oh