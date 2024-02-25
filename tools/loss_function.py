import random
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from scipy.optimize import linear_sum_assignment
from torch import Tensor


def reconstruction(input, recon):

    assert input.size() == recon.size(), "Different sizes!"

    return torch.sum((input - recon) ** 2) / input.shape[0]


def kl_divergence_reg(mu, logvar):
    """Regularizer for vae

    Args:
        mu (float): mean of distribution
        logvar (float): log(var) of distribution 
    """
    return torch.sum(-0.5*(1 + logvar - mu.pow(2) - logvar.exp()))


def match_cosine_similarity(z0, z1, neg_sample=3):
    """_summary_

    Args:
        z0 (Tensor): Feature / slot tensor
        z1 (Tensor): Feature / slot tensor
        neg_sample (int, optional): Number of pairwise different slots. Defaults to 3.
    """
    B = z0.shape[0]

    z0_norm = F.normalize(z0, dim=-1)  # shape: [batch, slots, slot_dim]
    z1_norm = F.normalize(z1, dim=-1)

    # shape: [batch, n_slots, n_slots]
    similarity = 1 - torch.einsum('bki,bki->bk', z0_norm, z1_norm)

    cos_sim, _ = torch.sort(similarity, descending=True)

    # for b in range(B):
    #    sim = similarity[b].clone().detach().cpu().numpy()
    #    row, col = linear_sum_assignment(sim)

    #    cos_sim += similarity[b][row, col][:-neg_sample]

    # pos_anchor = torch.sum(torch.exp(1 - similarity[b][row, col][:-neg_sample]))
    # neg_anchor = torch.sum(torch.exp(1 - similarity[b][row, col][-neg_sample:]))

    # softmax = (pos_anchor / (pos_anchor + neg_anchor))

    # cos_sim += -torch.log(softmax)

    return torch.sum(cos_sim[:, :-neg_sample])


def contrastive_z(z0: Tensor, z1: Tensor, temperature=1):
    """Not used currently!
    This function creates a contrastive loss given an predicted mask

    Args:
        z0 (Tensor): Shape: [batch_size, height*width, num_features]
        z1 (Tensor): Shape: [batch_size, height*width, num_features]
    """

    B, HW, N = z0.size()

    z0_norm = F.normalize(z0, dim=-1)
    z1_norm = F.normalize(z1, dim=-1)

    # shape: [batch, height*width]
    similarities = torch.einsum('bji,bji->bj', z0_norm, z1_norm)

    delta_max = torch.max(similarities, dim=1).values
    delta_min = torch.min(similarities, dim=1).values

    with torch.no_grad():
        tau = ((delta_max - delta_min) / 2).unsqueeze(-1).expand(B, HW)

        mask = (similarities > tau).int()  # shape: [batch, hw]

    return torch.mean(.5 * ((1 - mask) * (similarities) + mask * torch.maximum(torch.Tensor([0.]).cuda(), temperature - similarities)))


def get_bounding_box(segmentation: Tensor) -> Tuple[List[Tensor], Tensor]:
    """This function creates bounding boxes around given object 
    Args:
        segmentation (Tensor): Predicted object mask
    Returns:
        Tuple[List[Tensor], Tensor]: List of bounding box corners and object index
    """

    # shape = [H, W]
    unique = torch.unique(segmentation)
    num_objects = len(unique)

    bounding_boxes = torch.zeros(
        (num_objects, 4), device=segmentation.device, dtype=torch.int)

    for index, val in enumerate(unique):

        bin_mask = (segmentation == val)

        y, x = torch.where(bin_mask != 0)

        bounding_boxes[index, 0] = torch.min(x)
        bounding_boxes[index, 1] = torch.min(y)
        bounding_boxes[index, 2] = torch.max(x)
        bounding_boxes[index, 3] = torch.max(y)

    return bounding_boxes, unique


def get_mask_indices(bbox: Tensor, image_size: int, patch_size: int) -> List[List[List[int]]]:

    levels = np.arange(patch_size, image_size+patch_size, patch_size)

    mask_indices = []  # List[List[Tensors]]

    # iterate over all masks in batch
    for i in range(len(bbox)):

        bbox_idx = bbox[i]

        # size of bbox
        diff = (bbox_idx[:, 2:4] - bbox_idx[:, 0:2])

        num_blocks = (diff // patch_size)

        # check size
        sufficient_size = (num_blocks > 0)

        sufficient_size = torch.logical_and(
            sufficient_size[:, 0], sufficient_size[:, 1])

        bbox_sufficient = bbox_idx[sufficient_size].to("cpu").numpy()

        bbox_discrete = np.digitize(bbox_sufficient, levels)

        bbox_indices = []
        # iterate over all objects in mask
        for j in range(bbox_discrete.shape[0]):
            x_min, y_min, x_max, y_max = bbox_discrete[j]
            indices = [
                x*patch_size + y for x in range(x_min, x_max) for y in range(y_min, y_max)]
            bbox_indices.append(indices)

        mask_indices.append(bbox_indices)

    return mask_indices


def rand_permute(size: int, ratio: int = 0.5) -> Tuple[Tensor, Tensor]:
    """Return masked indices

    Args:
        size (int): Number of patches
        ratio (int, optional): Mask ratio. Defaults to 0.5.

    Returns:
        Tuple[Tensor, Tensor]: Masked patch indices
    """

    if size == 1:
        return torch.zeros(1, dtype=torch.int32), torch.zeros(1, dtype=torch.int32)

    noise = torch.rand(size)
    # ascend: small is keep, large is remove
    ids_shuffle = torch.argsort(noise).type(torch.int32)

    keep = int(size * ratio)

    return ids_shuffle[:keep], ids_shuffle[keep:]


def contrastive_mask(mask: Tensor, features: Tensor, threshold: float = 0.01, tau: float = 1.):
    """This function creates a contrastive loss given an predicted mask

    Args:
        mask (Tensor): Shape: [batch, slots, channels, height, width]
        features (Tensor): [batch, channels, height, width]
        threshold (float): Overlap threshold for considering neighbors
    """
    B, S, _, H, W = mask.shape
    _, C, _, _ = features.shape

    mask = mask.softmax(dim=1)
    alpha = mask.argmax(dim=1).squeeze(1).detach()

    with torch.no_grad():
        mask_sum = torch.sum(mask, dim=[-1, -2, -3])

    device = mask.device

    # get neighbouring slots

    # calculate mask overlap
    with torch.no_grad():
        overlap = torch.einsum("bscwh,bzcwh->bsz", mask, mask)

        union = list()
        for b in range(B):
            overlap[b].fill_diagonal_(0)

            x, y = torch.meshgrid(mask_sum[b], mask_sum[b])
            union.append(x + y)
        union = torch.stack(union)

        overlap /= (union + 1e-9)

        condition = (overlap >= 5e-3)

    # condition = condition & (torch.sum(condition, dim=1, keepdim=True) > 3).expand(B, S, S).permute(0, 2, 1)

    # get masked features
    mask = mask.squeeze(2)
    masked_features = torch.einsum("bshw, bfhw-> bsfhw", mask, features)

    patch_size = 16
    feature_patches = rearrange(
        masked_features, 'b s c (h s1) (w s2) -> b s (s1 s2) c (h w)', s1=patch_size, s2=patch_size)

    # pool and normalize featues
    # [batch, slots, features]
    masked_features = torch.mean(masked_features, dim=[-1, -2])
    masked_features = F.normalize(masked_features, dim=-1)  # [batch, slots]

    loss = torch.Tensor([0.]).to(device)

    bbox = list()
    unique = list()

    for i in range(B):
        # per image
        bbox_i, unique_i = get_bounding_box(alpha[i])
        bbox.append(bbox_i)
        unique.append(unique_i)

    mask_indices = get_mask_indices(bbox, alpha.size()[1], 16)

    for b in range(B):

        num_objects = len(mask_indices[b])

        for n in range(num_objects):

            # object patches
            obj_patches = mask_indices[b][n]
            slot_id = unique[b][n]

            indices = condition[b, slot_id].nonzero()

            ids_keep, ids_hide = rand_permute(len(obj_patches))

            ids_keep = torch.index_select(
                torch.LongTensor(obj_patches), 0, ids_keep)
            ids_hide = torch.index_select(
                torch.LongTensor(obj_patches), 0, ids_hide)

            pos_sample1 = \
                torch.mean(torch.index_select(
                    feature_patches[b, slot_id], 0, ids_keep.to(device)), dim=[0, -1])
            pos_sample2 = \
                torch.mean(torch.index_select(
                    feature_patches[b, slot_id], 0, ids_hide.to(device)), dim=[0, -1])

            pos_sample1 = F.normalize(pos_sample1.unsqueeze(0))
            pos_sample2 = F.normalize(pos_sample2.unsqueeze(0))

            pos_sample = torch.sum(pos_sample1 * pos_sample2).unsqueeze(0)

            # get negative samples
            neg_samples = (
                masked_features[b, slot_id] * masked_features[b, indices.flatten()]).sum(dim=1)

            logits = torch.cat((pos_sample, neg_samples)).unsqueeze(0)
            labels = torch.zeros(1, dtype=torch.int64).to(device)

            loss += F.cross_entropy(logits, labels)

    return loss
