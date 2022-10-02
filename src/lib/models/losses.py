# ------------------------------------------------------------------------------
# Portions of this code are from
# CornerNet (https://github.com/princeton-vl/CornerNet)
# Copyright (c) 2018, University of Michigan
# Licensed under the BSD 3-Clause License
# ------------------------------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
from .utils import _tranpose_and_gather_feat
import torch.nn.functional as F
import numpy as np


def _slow_neg_loss(pred, gt):
    '''focal loss from CornerNet'''
    pos_inds = gt.eq(1)
    neg_inds = gt.lt(1)

    neg_weights = torch.pow(1 - gt[neg_inds], 4)

    loss = 0
    pos_pred = pred[pos_inds]
    neg_pred = pred[neg_inds]

    pos_loss = torch.log(pos_pred) * torch.pow(1 - pos_pred, 2)
    neg_loss = torch.log(1 - neg_pred) * torch.pow(neg_pred, 2) * neg_weights

    num_pos = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    if pos_pred.nelement() == 0:
        loss = loss - neg_loss
    else:
        loss = loss - (pos_loss + neg_loss) / num_pos
    return loss


def _neg_loss(pred, gt):
    """ Modified focal loss. Exactly the same as CornerNet.
      Runs faster and costs a little bit more memory
    Arguments:
      pred (batch x c x h x w)
      gt_regr (batch x c x h x w)
  """
    pos_inds = gt.eq(1).float()
    neg_inds = gt.lt(1).float()

    neg_weights = torch.pow(1 - gt, 4)

    loss = 0

    pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
    neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds

    num_pos = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    if num_pos == 0:
        loss = loss - neg_loss
    else:
        loss = loss - (pos_loss + neg_loss) / num_pos
    return loss


def _not_faster_neg_loss(pred, gt):
    pos_inds = gt.eq(1).float()
    neg_inds = gt.lt(1).float()
    num_pos = pos_inds.float().sum()
    neg_weights = torch.pow(1 - gt, 4)

    loss = 0
    trans_pred = pred * neg_inds + (1 - pred) * pos_inds
    weight = neg_weights * neg_inds + pos_inds
    all_loss = torch.log(1 - trans_pred) * torch.pow(trans_pred, 2) * weight
    all_loss = all_loss.sum()

    if num_pos > 0:
        all_loss /= num_pos
    loss -= all_loss
    return loss


def _slow_reg_loss(regr, gt_regr, mask):
    num = mask.float().sum()
    mask = mask.unsqueeze(2).expand_as(gt_regr)

    regr = regr[mask]
    gt_regr = gt_regr[mask]

    regr_loss = nn.functional.smooth_l1_loss(regr, gt_regr, reduction='sum')
    regr_loss = regr_loss / (num + 1e-4)
    return regr_loss


def _reg_loss(regr, gt_regr, mask):
    """ L1 regression loss
    Arguments:
      regr (batch x max_objects x dim)
      gt_regr (batch x max_objects x dim)
      mask (batch x max_objects)
  """
    num = mask.float().sum()
    mask = mask.unsqueeze(2).expand_as(gt_regr).float()

    regr = regr * mask
    gt_regr = gt_regr * mask

    regr_loss = nn.functional.smooth_l1_loss(regr, gt_regr, reduction='sum')
    regr_loss = regr_loss / (num + 1e-4)
    return regr_loss


class FocalLoss(nn.Module):
    """nn.Module warpper for focal loss"""

    def __init__(self):
        super(FocalLoss, self).__init__()
        self.neg_loss = _neg_loss

    def forward(self, out, target):
        return self.neg_loss(out, target)


class GiouLoss(nn.Module):
    def __int__(self):
        super(GiouLoss, self).__int__()

    # def forward(self, preds, weight, bbox, eps=1e-10, iou_weight=1.):
    #     """
    #     (focal) EIOU Loss
    #     :param preds:[[x1,y1,x2,y2], [x1,y1,x2,y2],,,]
    #     :param bbox:[[x1,y1,x2,y2], [x1,y1,x2,y2],,,]
    #     :return: loss
    #     """
    #     pos_mask = weight > 0
    #     # avg_factor = torch.sum(pos_mask).float().item() + 1e-4
    #     preds = preds[pos_mask].view(-1, 4)
    #     bbox = bbox[pos_mask].view(-1, 4)
    #     # print('#' * 100)
    #     # print(preds[-3:])
    #     # print('-' * 100)
    #     # print(bbox[-3:])
    #
    #     ix1 = torch.max(preds[:, 0], bbox[:, 0])
    #     iy1 = torch.max(preds[:, 1], bbox[:, 1])
    #     ix2 = torch.min(preds[:, 2], bbox[:, 2])
    #     iy2 = torch.min(preds[:, 3], bbox[:, 3])
    #
    #     iw = (ix2 - ix1 + 1.0).clamp(min=0.)
    #     ih = (iy2 - iy1 + 1.0).clamp(min=0.)
    #
    #     w = preds[:, 2] - preds[:, 0]
    #     h = preds[:, 3] - preds[:, 1]
    #     wg = bbox[:, 2] - bbox[:, 0]
    #     hg = bbox[:, 3] - bbox[:, 1]
    #
    #     # overlaps
    #     inters = iw * ih
    #
    #     # union
    #     uni = (preds[:, 2] - preds[:, 0] + 1.0) * (preds[:, 3] - preds[:, 1] + 1.0) + (
    #                 bbox[:, 2] - bbox[:, 0] + 1.0) * (
    #                   bbox[:, 3] - bbox[:, 1] + 1.0) - inters
    #
    #     # iou
    #     iou = inters / (uni + eps)
    #
    #     # inter_diag
    #     cxpreds = (preds[:, 2] + preds[:, 0]) / 2
    #     cypreds = (preds[:, 3] + preds[:, 1]) / 2
    #
    #     cxbbox = (bbox[:, 2] + bbox[:, 0]) / 2
    #     cybbox = (bbox[:, 3] + bbox[:, 1]) / 2
    #
    #     inter_diag = (cxbbox - cxpreds) ** 2 + (cybbox - cypreds) ** 2
    #
    #     # outer_diag
    #     ox1 = torch.min(preds[:, 0], bbox[:, 0])
    #     oy1 = torch.min(preds[:, 1], bbox[:, 1])
    #     ox2 = torch.max(preds[:, 2], bbox[:, 2])
    #     oy2 = torch.max(preds[:, 3], bbox[:, 3])
    #
    #     outer_diag = (ox1 - ox2) ** 2 + (oy1 - oy2) ** 2 + eps
    #
    #     diou_term = inter_diag / outer_diag
    #     diou_term = torch.clamp(diou_term, min=0., max=1.0)
    #
    #     # EIOU term
    #     c2_w = (ox2 - ox1) ** 2 + eps
    #     c2_h = (oy2 - oy1) ** 2 + eps
    #     rho2_w = (w - wg) ** 2
    #     rho2_h = (h - hg) ** 2
    #     eiou_term = (rho2_w / c2_w) + (rho2_h / c2_h)
    #
    #     # Focal-EIOU
    #     eiou = torch.mean((1 - iou + diou_term + eiou_term) * iou_weight)
    #     # print(eiou)
    #     # focal_eiou有问题，loss是nan
    #     # focal_eiou = torch.mean(iou ** 0.5 * eiou)
    #     # print(focal_eiou)
    #     return eiou

    # def forward(self, preds, weight, bbox, eps=1e-7, reduction='mean'):
    #     """
    #     DIOU Loss
    #     :param preds:[[x1,y1,x2,y2], [x1,y1,x2,y2],,,]
    #     :param bbox:[[x1,y1,x2,y2], [x1,y1,x2,y2],,,]
    #     :return: loss
    #     """
    #     pos_mask = weight > 0
    #     # avg_factor = torch.sum(pos_mask).float().item() + 1e-4
    #     preds = preds[pos_mask].view(-1, 4)
    #     bbox = bbox[pos_mask].view(-1, 4)
    #     print('#' * 100)
    #     print(preds)
    #     print('-' * 100)
    #     print(bbox)
    #     ix1 = torch.max(preds[:, 0], bbox[:, 0])
    #     iy1 = torch.max(preds[:, 1], bbox[:, 1])
    #     ix2 = torch.min(preds[:, 2], bbox[:, 2])
    #     iy2 = torch.min(preds[:, 3], bbox[:, 3])
    #
    #     iw = (ix2 - ix1 + 1.0).clamp(min=0.)
    #     ih = (iy2 - iy1 + 1.0).clamp(min=0.)
    #
    #     # overlaps
    #     inters = iw * ih
    #
    #     # union
    #     uni = (preds[:, 2] - preds[:, 0] + 1.0) * (preds[:, 3] - preds[:, 1] + 1.0) + (
    #                 bbox[:, 2] - bbox[:, 0] + 1.0) * (
    #                   bbox[:, 3] - bbox[:, 1] + 1.0) - inters
    #
    #     # iou
    #     iou = inters / (uni + eps)
    #
    #     # inter_diag
    #     cxpreds = (preds[:, 2] + preds[:, 0]) / 2
    #     cypreds = (preds[:, 3] + preds[:, 1]) / 2
    #
    #     cxbbox = (bbox[:, 2] + bbox[:, 0]) / 2
    #     cybbox = (bbox[:, 3] + bbox[:, 1]) / 2
    #
    #     inter_diag = (cxbbox - cxpreds) ** 2 + (cybbox - cypreds) ** 2
    #
    #     # outer_diag
    #     ox1 = torch.min(preds[:, 0], bbox[:, 0])
    #     oy1 = torch.min(preds[:, 1], bbox[:, 1])
    #     ox2 = torch.max(preds[:, 2], bbox[:, 2])
    #     oy2 = torch.max(preds[:, 3], bbox[:, 3])
    #
    #     outer_diag = (ox1 - ox2) ** 2 + (oy1 - oy2) ** 2 + eps
    #
    #     diou = iou - inter_diag / outer_diag
    #     diou = torch.clamp(diou, min=-1.0, max=1.0)
    #
    #     diou_loss = 1 - diou
    #
    #     if reduction == 'mean':
    #         loss = torch.mean(diou_loss)
    #     elif reduction == 'sum':
    #         loss = torch.sum(diou_loss)
    #     else:
    #         raise NotImplementedError
    #     return loss

    # def forward(self, preds, weight, bbox, eps=1e-9, reduction='mean'):
    #     """
    #     IOU Loss
    #     :param preds:[[x1,y1,x2,y2], [x1,y1,x2,y2],,,]
    #     :param bbox:[[x1,y1,x2,y2], [x1,y1,x2,y2],,,]
    #     :return: loss
    #     """
    #     pos_mask = weight > 0
    #     # avg_factor = torch.sum(pos_mask).float().item() + 1e-4
    #     preds = preds[pos_mask].view(-1, 4)
    #     bbox = bbox[pos_mask].view(-1, 4)
    #     print('#' * 100)
    #     print(preds[-1])
    #     print('-' * 100)
    #     print(bbox[-1])
    #     x1 = torch.max(preds[:, 0], bbox[:, 0])
    #     y1 = torch.max(preds[:, 1], bbox[:, 1])
    #     x2 = torch.min(preds[:, 2], bbox[:, 2])
    #     y2 = torch.min(preds[:, 3], bbox[:, 3])
    #
    #     w = (x2 - x1 + 1.0).clamp(0.)
    #     h = (y2 - y1 + 1.0).clamp(0.)
    #
    #     inters = w * h
    #
    #     uni = (preds[:, 2] - preds[:, 0] + 1.0) * (preds[:, 3] - preds[:, 1] + 1.0) + (
    #                 bbox[:, 2] - bbox[:, 0] + 1.0) * (
    #                   bbox[:, 3] - bbox[:, 1] + 1.0) - inters
    #
    #     ious = (inters / uni).clamp(min=eps)
    #     loss = -ious.log()
    #
    #     if reduction == 'mean':
    #         loss = torch.mean(loss)
    #     elif reduction == 'sum':
    #         loss = torch.sum(loss)
    #     else:
    #         raise NotImplementedError
    #     return loss

    # def forward(self, preds, weight, bbox, eps=1e-7, reduction='mean'):
    #     """
    #    https://github.com/sfzhang15/ATSS/blob/master/atss_core/modeling/rpn/atss/loss.py#L36
    #     :param preds:[[x1,y1,x2,y2], [x1,y1,x2,y2],,,]
    #     :param bbox:[[x1,y1,x2,y2], [x1,y1,x2,y2],,,]
    #     :return: loss
    #     """
    #     pos_mask = weight > 1e-3
    #     # avg_factor = torch.sum(pos_mask).float().item() + 1e-4
    #     preds = preds[pos_mask].view(-1, 4)
    #     bbox = bbox[pos_mask].view(-1, 4)
    #     print('#' * 100)
    #     print(preds)
    #     print('-' * 100)
    #     print(bbox)
    #     ix1 = torch.max(preds[:, 0], bbox[:, 0])
    #     iy1 = torch.max(preds[:, 1], bbox[:, 1])
    #     ix2 = torch.min(preds[:, 2], bbox[:, 2])
    #     iy2 = torch.min(preds[:, 3], bbox[:, 3])
    #
    #     iw = (ix2 - ix1 + 1.0).clamp(0.)
    #     ih = (iy2 - iy1 + 1.0).clamp(0.)
    #
    #     # overlap
    #     inters = iw * ih
    #
    #     # union
    #     uni = (preds[:, 2] - preds[:, 0] + 1.0) * (preds[:, 3] - preds[:, 1] + 1.0) + (
    #                 bbox[:, 2] - bbox[:, 0] + 1.0) * (
    #                   bbox[:, 3] - bbox[:, 1] + 1.0) - inters + eps
    #
    #     # ious
    #     ious = inters / uni
    #
    #     ex1 = torch.min(preds[:, 0], bbox[:, 0])
    #     ey1 = torch.min(preds[:, 1], bbox[:, 1])
    #     ex2 = torch.max(preds[:, 2], bbox[:, 2])
    #     ey2 = torch.max(preds[:, 3], bbox[:, 3])
    #     ew = (ex2 - ex1 + 1.0).clamp(min=0.)
    #     eh = (ey2 - ey1 + 1.0).clamp(min=0.)
    #
    #     # enclose erea
    #     enclose = ew * eh + eps
    #
    #     giou = ious - (enclose - uni) / enclose
    #
    #     loss = 1 - giou
    #
    #     if reduction == 'mean':
    #         loss = torch.mean(loss)
    #     elif reduction == 'sum':
    #         loss = torch.sum(loss)
    #     else:
    #         raise NotImplementedError
    #     return loss


    def forward(self, pred, weight, target, eps = 1e-10):
        """
        Computing the GIoU loss between a set of predicted bboxes and target bboxes.
        Arguments:
              output (batch x dim x h x w)      pred/target (batch, h, w, 4)
              mask (batch x max_objects)        weight (batch × 1, h, w)
              ind (batch x max_objects)
              target (batch x max_objects x dim)
        """
        pos_mask = weight > 0
        weight = weight[pos_mask]
        avg_factor = weight.sum() + eps
        # avg_factor = torch.sum(pos_mask).float().item() + 1e-4
        bboxes1 = pred[pos_mask].view(-1, 4)
        bboxes2 = target[pos_mask].view(-1, 4)
        # print('#' * 100)
        # print(bboxes1)
        # print('-' * 100)
        # print(bboxes2)

        lt = torch.max(bboxes1[:, :2], bboxes2[:, :2])  # [rows, 2]
        rb = torch.min(bboxes1[:, 2:], bboxes2[:, 2:])  # [rows, 2]
        wh = (rb - lt).clamp(min=0)  # [rows, 2]
        enclose_x1y1 = torch.min(bboxes1[:, :2], bboxes2[:, :2])
        enclose_x2y2 = torch.max(bboxes1[:, 2:], bboxes2[:, 2:])
        enclose_wh = (enclose_x2y2 - enclose_x1y1).clamp(min=0)

        overlap = wh[:, 0] * wh[:, 1]
        ap = (bboxes1[:, 2] - bboxes1[:, 0]) * (bboxes1[:, 3] - bboxes1[:, 1])
        ag = (bboxes2[:, 2] - bboxes2[:, 0]) * (bboxes2[:, 3] - bboxes2[:, 1])
        ious = overlap / (ap + ag - overlap + eps)

        enclose_area = enclose_wh[:, 0] * enclose_wh[:, 1] + eps    # i.e. C in paper
        u = ap + ag - overlap
        gious = ious - (enclose_area - u) / enclose_area
        iou_distances = 1 - gious

        np.set_printoptions(threshold=np.inf)
        print(weight.shape)

        print('#' * 100)
        print(iou_distances.cpu().detach().numpy(), iou_distances.shape)
        # return torch.sum(iou_distances) / avg_factor
        return torch.sum(iou_distances * weight)[None] / avg_factor


class RegLoss(nn.Module):
    """Regression loss for an output tensor
    Arguments:
      output (batch x dim x h x w)
      mask (batch x max_objects)
      ind (batch x max_objects)
      target (batch x max_objects x dim)
  """

    def __init__(self):
        super(RegLoss, self).__init__()

    def forward(self, output, mask, ind, target):
        pred = _tranpose_and_gather_feat(output, ind)
        loss = _reg_loss(pred, target, mask)
        return loss


class RegL1Loss(nn.Module):
    def __init__(self):
        super(RegL1Loss, self).__init__()

    def forward(self, output, mask, ind, target):
        pred = _tranpose_and_gather_feat(output, ind)
        mask = mask.unsqueeze(2).expand_as(pred).float()
        # loss = F.l1_loss(pred * mask, target * mask, reduction='elementwise_mean')
        loss = F.l1_loss(pred * mask, target * mask, reduction='sum')
        loss = loss / (mask.sum() + 1e-4)
        return loss


class NormRegL1Loss(nn.Module):
    def __init__(self):
        super(NormRegL1Loss, self).__init__()

    def forward(self, output, mask, ind, target):
        pred = _tranpose_and_gather_feat(output, ind)
        mask = mask.unsqueeze(2).expand_as(pred).float()
        # loss = F.l1_loss(pred * mask, target * mask, reduction='elementwise_mean')
        pred = pred / (target + 1e-4)
        target = target * 0 + 1
        loss = F.l1_loss(pred * mask, target * mask, reduction='sum')
        loss = loss / (mask.sum() + 1e-4)
        return loss


class RegWeightedL1Loss(nn.Module):
    def __init__(self):
        super(RegWeightedL1Loss, self).__init__()

    def forward(self, output, mask, ind, target):
        pred = _tranpose_and_gather_feat(output, ind)
        mask = mask.float()
        # loss = F.l1_loss(pred * mask, target * mask, reduction='elementwise_mean')
        loss = F.l1_loss(pred * mask, target * mask, reduction='sum')
        loss = loss / (mask.sum() + 1e-4)
        return loss


class L1Loss(nn.Module):
    def __init__(self):
        super(L1Loss, self).__init__()

    def forward(self, output, mask, ind, target):
        pred = _tranpose_and_gather_feat(output, ind)
        mask = mask.unsqueeze(2).expand_as(pred).float()
        loss = F.l1_loss(pred * mask, target * mask, reduction='elementwise_mean')
        return loss


class BinRotLoss(nn.Module):
    def __init__(self):
        super(BinRotLoss, self).__init__()

    def forward(self, output, mask, ind, rotbin, rotres):
        pred = _tranpose_and_gather_feat(output, ind)
        loss = compute_rot_loss(pred, rotbin, rotres, mask)
        return loss


def compute_res_loss(output, target):
    return F.smooth_l1_loss(output, target, reduction='elementwise_mean')


# TODO: weight
def compute_bin_loss(output, target, mask):
    mask = mask.expand_as(output)
    output = output * mask.float()
    return F.cross_entropy(output, target, reduction='elementwise_mean')


def compute_rot_loss(output, target_bin, target_res, mask):
    # output: (B, 128, 8) [bin1_cls[0], bin1_cls[1], bin1_sin, bin1_cos, 
    #                 bin2_cls[0], bin2_cls[1], bin2_sin, bin2_cos]
    # target_bin: (B, 128, 2) [bin1_cls, bin2_cls]
    # target_res: (B, 128, 2) [bin1_res, bin2_res]
    # mask: (B, 128, 1)
    # import pdb; pdb.set_trace()
    output = output.view(-1, 8)
    target_bin = target_bin.view(-1, 2)
    target_res = target_res.view(-1, 2)
    mask = mask.view(-1, 1)
    loss_bin1 = compute_bin_loss(output[:, 0:2], target_bin[:, 0], mask)
    loss_bin2 = compute_bin_loss(output[:, 4:6], target_bin[:, 1], mask)
    loss_res = torch.zeros_like(loss_bin1)
    if target_bin[:, 0].nonzero().shape[0] > 0:
        idx1 = target_bin[:, 0].nonzero()[:, 0]
        valid_output1 = torch.index_select(output, 0, idx1.long())
        valid_target_res1 = torch.index_select(target_res, 0, idx1.long())
        loss_sin1 = compute_res_loss(
            valid_output1[:, 2], torch.sin(valid_target_res1[:, 0]))
        loss_cos1 = compute_res_loss(
            valid_output1[:, 3], torch.cos(valid_target_res1[:, 0]))
        loss_res += loss_sin1 + loss_cos1
    if target_bin[:, 1].nonzero().shape[0] > 0:
        idx2 = target_bin[:, 1].nonzero()[:, 0]
        valid_output2 = torch.index_select(output, 0, idx2.long())
        valid_target_res2 = torch.index_select(target_res, 0, idx2.long())
        loss_sin2 = compute_res_loss(
            valid_output2[:, 6], torch.sin(valid_target_res2[:, 1]))
        loss_cos2 = compute_res_loss(
            valid_output2[:, 7], torch.cos(valid_target_res2[:, 1]))
        loss_res += loss_sin2 + loss_cos2
    return loss_bin1 + loss_bin2 + loss_res


class TripletLoss(nn.Module):
    """Triplet loss with hard positive/negative mining.
    Reference:
    Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.
    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/loss/triplet.py.
    Args:
        margin (float): margin for triplet.
    """

    def __init__(self, margin=0.3, mutual_flag=False):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        self.mutual = mutual_flag

    def forward(self, inputs, targets):
        """
        Args:
            inputs: feature matrix with shape (batch_size, feat_dim)
            targets: ground truth labels with shape (num_classes)
        """
        n = inputs.size(0)
        # inputs = 1. * inputs / (torch.norm(inputs, 2, dim=-1, keepdim=True).expand_as(inputs) + 1e-12)
        # Compute pairwise distance, replace by the official when merged
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(1, -2, inputs, inputs.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
            dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)
        # Compute ranking hinge loss
        y = torch.ones_like(dist_an)
        loss = self.ranking_loss(dist_an, dist_ap, y)
        if self.mutual:
            return loss, dist
        return loss


def reduce_loss(loss, reduction):
    """Reduce loss as specified.

    Args:
        loss (Tensor): Elementwise loss tensor.
        reduction (str): Options are "none", "mean" and "sum".

    Return:
        Tensor: Reduced loss tensor.
    """
    reduction_enum = F._Reduction.get_enum(reduction)
    # none: 0, elementwise_mean:1, sum: 2
    if reduction_enum == 0:
        return loss
    elif reduction_enum == 1:
        return loss.mean()
    elif reduction_enum == 2:
        return loss.sum()


def weight_reduce_loss(loss, weight=None, reduction='mean', avg_factor=None):
    """Apply element-wise weight and reduce loss.
    Args:
        loss (Tensor): Element-wise loss.
        weight (Tensor): Element-wise weights.
        reduction (str): Same as built-in losses of PyTorch.
        avg_factor (float): Avarage factor when computing the mean of losses.

    Returns:
        Tensor: Processed loss values.
    """
    # if weight is specified, apply element-wise weight
    if weight is not None:
        loss = loss * weight

    # if avg_factor is not specified, just reduce the loss
    if avg_factor is None:
        loss = reduce_loss(loss, reduction)
    else:
        # if reduction is mean, then average the loss by avg_factor
        if reduction == 'mean':
            loss = loss.sum() / avg_factor
        # if reduction is 'none', then do nothing, otherwise raise an error
        elif reduction != 'none':
            raise ValueError('avg_factor can not be used with reduction="sum"')
    return loss


def _varifocal_loss(pred,
                    target,
                    weight=None,
                    alpha=0.75,
                    gamma=2.0,
                    iou_weighted=True,
                    reduction='mean',
                    avg_factor=None):
    """`Varifocal Loss <https://arxiv.org/abs/2008.13367>`_
    Args:
        pred (torch.Tensor): The prediction with shape (N, C), C is the
            number of classes
        target (torch.Tensor): The learning target of the iou-aware
            classification score with shape (N, C), C is the number of classes.
        weight (torch.Tensor, optional): The weight of loss for each
            prediction. Defaults to None.
        alpha (float, optional): A balance factor for the negative part of
            Varifocal Loss, which is different from the alpha of Focal Loss.
            Defaults to 0.75.
        gamma (float, optional): The gamma for calculating the modulating
            factor. Defaults to 2.0.
        iou_weighted (bool, optional): Whether to weight the loss of the
            positive example with the iou target. Defaults to True.
        reduction (str, optional): The method used to reduce the loss into
            a scalar. Defaults to 'mean'. Options are "none", "mean" and
            "sum".
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
    """
    # pred and target should be of the same size
    assert pred.size() == target.size()
    pred_sigmoid = pred     # pred.sigmoid() 此处不需要，在mot.py里执行过
    target = target.type_as(pred)
    if iou_weighted:
        focal_weight = target * (target > 0.0).float() + \
                       alpha * (pred_sigmoid - target).abs().pow(gamma) * \
                       (target <= 0.0).float()
    else:
        focal_weight = (target > 0.0).float() + \
                       alpha * (pred_sigmoid - target).abs().pow(gamma) * \
                       (target <= 0.0).float()
    loss = F.binary_cross_entropy_with_logits(
        pred, target, reduction='none') * focal_weight
    loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
    return loss


class VarifocalLoss(nn.Module):
    """`Varifocal Loss <https://arxiv.org/abs/2008.13367>`_
    Args:
        use_sigmoid (bool, optional): Whether the prediction is
            used for sigmoid or softmax. Defaults to True.
        alpha (float, optional): A balance factor for the negative part of
            Varifocal Loss, which is different from the alpha of Focal
            Loss. Defaults to 0.75.
        gamma (float, optional): The gamma for calculating the modulating
            factor. Defaults to 2.0.
        iou_weighted (bool, optional): Whether to weight the loss of the
            positive examples with the iou target. Defaults to True.
        reduction (str, optional): The method used to reduce the loss into
            a scalar. Defaults to 'mean'. Options are "none", "mean" and
            "sum".
        loss_weight (float, optional): Weight of loss. Defaults to 1.0.
    """
    def __init__(self,
                 use_sigmoid=True,
                 alpha=0.75,
                 gamma=2.0,
                 iou_weighted=True,
                 reduction='mean',
                 loss_weight=1.0):
        super(VarifocalLoss, self).__init__()
        assert use_sigmoid is True, \
            'Only sigmoid varifocal loss supported now.'
        assert alpha >= 0.0
        self.use_sigmoid = use_sigmoid
        self.alpha = alpha
        self.gamma = gamma
        self.iou_weighted = iou_weighted
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        """Forward function.
        Args:
            pred (torch.Tensor): The prediction.
            target (torch.Tensor): The learning target of the prediction.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Options are "none", "mean" and "sum".
        Returns:
            torch.Tensor: The calculated loss
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if self.use_sigmoid:
            loss_cls = self.loss_weight * _varifocal_loss(
                pred,
                target,
                weight,
                alpha=self.alpha,
                gamma=self.gamma,
                iou_weighted=self.iou_weighted,
                reduction=reduction,
                avg_factor=avg_factor)
        else:
            raise NotImplementedError
        return loss_cls
