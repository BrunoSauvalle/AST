
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score

from MF_config import args


# imported from CLEVRTEX github
def reindex(tensor, reindex_tensor, dim=1):
    """
    Reindexes tensor along <dim> using reindex_tensor.
    Effectivelly permutes <dim> for each dimensions <dim based on values in reindex_tensor
    """
    # add dims at the end to match tensor dims.
    alignment_index = reindex_tensor.view(*reindex_tensor.shape,
                                          *([1] * (tensor.dim() - reindex_tensor.dim())))
    return torch.gather(tensor, dim, alignment_index.expand_as(tensor))


def ious_alignment(pred_masks, true_masks):
    tspec = dict(device=pred_masks.device)
    iou_matrix = torch.zeros(pred_masks.shape[0], pred_masks.shape[1], true_masks.shape[1], **tspec)

    true_masks_sums = true_masks.sum((-1, -2, -3))
    pred_masks_sums = pred_masks.sum((-1, -2, -3))

    pred_masks = pred_masks.to(torch.bool)
    true_masks = true_masks.to(torch.bool)

    # Fill IoU row-wise
    for pi in range(pred_masks.shape[1]):
        # Intersection against all cols
        # pandt = (pred_masks[:, pi:pi + 1] * true_masks).sum((-1, -2, -3))
        pandt = (pred_masks[:, pi:pi + 1] & true_masks).to(torch.float).sum((-1, -2, -3))
        # Union against all colls
        # port = pred_masks_sums[:, pi:pi + 1] + true_masks_sums
        port = (pred_masks[:, pi:pi + 1] | true_masks).to(torch.float).sum((-1, -2, -3))
        iou_matrix[:, pi] = pandt / port
        iou_matrix[pred_masks_sums[:, pi] == 0., pi] = 0.

    for ti in range(true_masks.shape[1]):
        iou_matrix[true_masks_sums[:, ti] == 0., :, ti] = 0.

    # NaNs, Inf might come from empty masks (sums are 0, such as on empty masks)
    # Set them to 0. as there are no intersections here and we should not reindex
    iou_matrix = torch.nan_to_num(iou_matrix, nan=0., posinf=0., neginf=0.)

    cost_matrix = iou_matrix.cpu().detach().numpy()
    ious = np.zeros(pred_masks.shape[:2])
    pred_inds = np.zeros(pred_masks.shape[:2], dtype=int)
    for bi in range(cost_matrix.shape[0]):
        true_ind, pred_ind = linear_sum_assignment(cost_matrix[bi].T, maximize=True)
        cost_matrix[bi].T[:, pred_ind].argmax(1)  # Gives which true mask is best for EACH predicted
        ious[bi] = cost_matrix[bi].T[true_ind, pred_ind]
        pred_inds[bi] = pred_ind

    ious = torch.from_numpy(ious).to(pred_masks.device)
    pred_inds = torch.from_numpy(pred_inds).to(pred_masks.device)
    return pred_inds, ious, iou_matrix

def compute_ari(pred_mask, true_mask, skip_0=False):
        B = pred_mask.shape[0]
        pm = pred_mask.argmax(axis=1).squeeze().view(B, -1).cpu().detach().numpy()
        tm = true_mask.argmax(axis=1).squeeze().view(B, -1).cpu().detach().numpy()
        aris = []
        for bi in range(B):
            t = tm[bi]
            p = pm[bi]
            if skip_0:
                p = p[t > 0]
                t = t[t > 0]
            ari_score = adjusted_rand_score(t, p)
            if ari_score != ari_score:
                print(f'NaN at bi')
            aris.append(ari_score)
        aris = torch.tensor(np.array(aris), device=pred_mask.device)
        return aris

# imported from GENESIS github


def iou_binary(mask_A, mask_B, debug=False):
    if debug:
        assert mask_A.shape == mask_B.shape
        assert mask_A.dtype == torch.bool
        assert mask_B.dtype == torch.bool
    intersection = (mask_A * mask_B).sum((1, 2, 3))
    union = (mask_A + mask_B).sum((1, 2, 3))
    # Return -100 if union is zero, else return IOU
    return torch.where(union == 0, torch.tensor(-100.0),
                       intersection.float() / union.float())

def average_segcover( mask_index,GT_mask_index, ignore_background=False):

    batch_size,_,_,h,w = mask_index.shape
    segA = GT_mask_index.reshape(batch_size,1,h,w).cpu()
    segB = mask_index.reshape(batch_size,1,h,w).cpu()
    """
    Covering of segA by segB
    segA.shape = [batch size, 1, img_dim1, img_dim2]
    segB.shape = [batch size, 1, img_dim1, img_dim2]
    scale: If true, take weighted mean over IOU values proportional to the
           the number of pixels of the mask being covered.
    Assumes labels in segA and segB are non-negative integers.
    Negative labels will be ignored.
    """

    assert segA.shape == segB.shape, f"{segA.shape} - {segB.shape}"
    assert segA.shape[1] == 1 and segB.shape[1] == 1
    bsz = segA.shape[0]
    nonignore = (segA >= 0)

    mean_scores = torch.tensor(bsz*[0.0])
    N = torch.tensor(bsz*[0])
    scaled_scores = torch.tensor(bsz*[0.0])
    scaling_sum = torch.tensor(bsz*[0])

    # Find unique label indices to iterate over
    if ignore_background:
        iter_segA = torch.unique(segA[segA > 0]).tolist()
    else:
        iter_segA = torch.unique(segA[segA >= 0]).tolist()
    iter_segB = torch.unique(segB[segB >= 0]).tolist()
    # Loop over segA
    for i in iter_segA:
        binaryA = segA == i
        if not binaryA.any():
            continue
        max_iou = torch.tensor(bsz*[0.0])
        # Loop over segB to find max IOU
        for j in iter_segB:
            # Do not penalise pixels that are in ignore regions
            binaryB = (segB == j) * nonignore
            if not binaryB.any():
                continue
            iou = iou_binary(binaryA, binaryB)
            max_iou = torch.where(iou > max_iou, iou, max_iou)
        # Accumulate scores
        mean_scores += max_iou
        N = torch.where(binaryA.sum((1, 2, 3)) > 0, N+1, N)
        scaled_scores += binaryA.sum((1, 2, 3)).float() * max_iou
        scaling_sum += binaryA.sum((1, 2, 3))

    # Compute coverage
    mean_sc = mean_scores / torch.max(N, torch.tensor(1)).float()
    scaled_sc = scaled_scores / torch.max(scaling_sum, torch.tensor(1)).float()

    # Sanity check
    assert (mean_sc >= 0).all() and (mean_sc <= 1).all(), mean_sc
    assert (scaled_sc >= 0).all() and (scaled_sc <= 1).all(), scaled_sc
    assert (mean_scores[N == 0] == 0).all()
    assert (mean_scores[nonignore.sum((1, 2, 3)) == 0] == 0).all()
    assert (scaled_scores[N == 0] == 0).all()
    assert (scaled_scores[nonignore.sum((1, 2, 3)) == 0] == 0).all()

    return mean_sc, scaled_sc

@torch.no_grad()
def evaluate(data,netE,netG, reduction=True, background_encoder = None, background_generator = None):
    h = args.image_height
    w = args.image_width
    training_mode = netE.training
    assert training_mode == netG.training

    netE.eval()
    netG.eval()

    input_images, background_images_with_error_prediction, GT_masks = data

    input_images = input_images.type(torch.cuda.FloatTensor).to(0)


    if background_encoder == None:
        background_images_with_error_prediction = background_images_with_error_prediction.type(
            torch.cuda.FloatTensor).to(args.device)
    else:
        background_training_mode = background_encoder.training
        background_encoder.eval()
        background_generator.eval()
        background_images_with_error_prediction = (1/255)*background_generator(background_encoder(255*input_images))
        if background_training_mode:
            background_encoder.train()
            background_generator.train()

    background_images = background_images_with_error_prediction[:, :3, :, :]
    latents = netE(input_images)
    rgb_images, foreground_masks, image_layers, activation_layers = netG(latents, background_images)

    batch_size = input_images.shape[0]
    max_set_size = args.max_set_size

    # number of active heads
    max_activation_per_layer_on_minibatch = torch.amax(activation_layers,  (1,2,3,4) ) # (K+1)
    number_of_active_heads = torch.sum(torch.ge(max_activation_per_layer_on_minibatch,1e-3), dim = 0).float() -1 # -1 is for background

    max_activation_per_layer_on_sample = torch.amax(activation_layers, (2, 3, 4))  # (K+1)N
    average_number_of_activated_heads = torch.mean(torch.sum(torch.ge(max_activation_per_layer_on_sample, 1e-3), dim=0).float()) -1

    #mse_loss
    mse_loss = nn.functional.mse_loss(input_images, rgb_images, reduction='none').sum((1, 2, 3))

    if torch.sum(GT_masks) == 0: # no GT masks, only mse loss can be computed and returned
        if training_mode:
            netE.train()
            netG.train()
        return torch.mean(mse_loss), 0, 0, 0, 0, 0, 0, 0,  number_of_active_heads, average_number_of_activated_heads

    GT_masks = GT_masks.type(torch.cuda.FloatTensor).to(
        args.device)
    activation_layers = activation_layers.reshape(max_set_size+1, batch_size, 1, h, w)  # K, N , H, W
    mask_index = torch.argmax(activation_layers, 0).expand(batch_size, 1, h, w).reshape(batch_size,1,1,h,w)  # n,1,1,H,W
    GT_mask_index = GT_masks.reshape(batch_size,1,1,h,w)
    pred_masks = (mask_index == torch.arange(max_set_size+1, device = args.device).view(1, max_set_size+1,1, 1, 1)).to(
            torch.float)
    true_masks = (GT_mask_index  == torch.arange(max_set_size+1, device = args.device).view(1, max_set_size+1, 1, 1,1)).to(
            torch.float)

    pred_reindex, ious, _ = ious_alignment(pred_masks, true_masks)
    pred_masks = reindex(pred_masks, pred_reindex, dim=1)
    truem = true_masks.any(-1).any(-1).any(-1)
    predm = pred_masks.any(-1).any(-1).any(-1)
    vism = truem | predm
    num_pairs = vism.to(torch.float).sum(-1)

    # mIoU
    mIoU = ious.sum(-1) / num_pairs

    #msc
    msc, scaled_sc = average_segcover(mask_index,GT_mask_index , ignore_background=False)
    msc_fg, scaled_sc_fg = average_segcover(mask_index,GT_mask_index , ignore_background=True)

    #ari
    ari = compute_ari(pred_masks,true_masks)
    ari_fg = compute_ari(pred_masks, true_masks, skip_0=True)


    if training_mode:
        netE.train()
        netG.train()

    if reduction:
        mse_loss = torch.mean(mse_loss)
        mIoU = torch.mean(mIoU)
        msc = torch.mean(msc)
        scaled_sc = torch.mean(scaled_sc)
        msc_fg = torch.mean(msc_fg)
        scaled_sc_fg = torch.mean(scaled_sc_fg)
        ari = torch.mean(ari)
        ari_fg = torch.mean(ari_fg)

    return mse_loss, mIoU, msc,scaled_sc,msc_fg,scaled_sc_fg, ari, ari_fg, number_of_active_heads, average_number_of_activated_heads
