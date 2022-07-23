from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
from .utils import _gather_feat, _tranpose_and_gather_feat


def _nms(heat, kernel=3):
    pad = (kernel - 1) // 2

    hmax = nn.functional.max_pool2d(
        heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()
    return heat * keep


def _topk_channel(scores, K=40):
    batch, cat, height, width = scores.size()

    topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)

    topk_inds = topk_inds % (height * width)
    topk_ys = torch.true_divide(topk_inds, width).int().float()
    topk_xs = (topk_inds % width).int().float()

    return topk_scores, topk_inds, topk_ys, topk_xs


def _topk(scores, K=40):
    """一共进行两次topk，第一次分别找出不同类里的前K个，然后在cat×K里再选出真正的前K个极大值"""
    batch, cat, height, width = scores.size()
    # batch * cat * K，batch代表batchsize，cat代表类别数，K代表K个最大值。
    topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)
    # index取值：[0, W x H - 1]
    topk_inds = topk_inds % (height * width)
    topk_ys = torch.true_divide(topk_inds, width).int().float()
    topk_xs = (topk_inds % width).int().float()
    # batch * K，index取值：[0, cat x K - 1]
    topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), K)
    topk_clses = torch.true_divide(topk_ind, K).int()
    topk_inds = _gather_feat(
        topk_inds.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_ys = _gather_feat(topk_ys.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_xs = _gather_feat(topk_xs.view(batch, -1, 1), topk_ind).view(batch, K)

    return topk_score, topk_inds, topk_clses, topk_ys, topk_xs


def mot_decode(heat, wh, reg=None, ltrb=False, K=100):
    batch, cat, height, width = heat.size()

    # heat = torch.sigmoid(heat)
    # perform nms on heatmaps   8-近邻极大值点
    heat = _nms(heat)

    scores, inds, clses, ys, xs = _topk(heat, K=K)
    if reg is not None:
        reg = _tranpose_and_gather_feat(reg, inds)
        reg = reg.view(batch, K, 2)
        xs = xs.view(batch, K, 1) + reg[:, :, 0:1]
        ys = ys.view(batch, K, 1) + reg[:, :, 1:2]
    else:
        xs = xs.view(batch, K, 1) + 0.5
        ys = ys.view(batch, K, 1) + 0.5
    wh = _tranpose_and_gather_feat(wh, inds)
    if ltrb:
        wh = wh.view(batch, K, 4)
    else:
        wh = wh.view(batch, K, 2)
    clses = clses.view(batch, K, 1).float()
    scores = scores.view(batch, K, 1)
    if ltrb:
        bboxes = torch.cat([xs - wh[..., 0:1],
                            ys - wh[..., 1:2],
                            xs + wh[..., 2:3],
                            ys + wh[..., 3:4]], dim=2)
        # box_indx1, box_indy1 = xs - wh[..., 0:1], ys - wh[..., 1:2]
        # box_indx2, box_indy2 = xs + wh[..., 2:3], ys + wh[..., 3:4]
    else:
        bboxes = torch.cat([xs - wh[..., 0:1] / 2,
                            ys - wh[..., 1:2] / 2,
                            xs + wh[..., 0:1] / 2,
                            ys + wh[..., 1:2] / 2], dim=2)
        # box_indx1, box_indy1 = xs - wh[..., 0:1] / 2, ys - wh[..., 1:2] / 2
        # box_indx2, box_indy2 = xs + wh[..., 2:3] / 2, ys + wh[..., 3:4] / 2
    detections = torch.cat([bboxes, scores, clses], dim=2)

    # ind_lt = (torch.mul(box_indy1, width) + box_indx1).squeeze(2).to(torch.int64).clamp(0, height*width - 1)
    # ind_rt = (torch.mul(box_indy1, width) + box_indx2).squeeze(2).to(torch.int64).clamp(0, height*width - 1)
    # ind_lb = (torch.mul(box_indy2, width) + box_indx1).squeeze(2).to(torch.int64).clamp(0, height*width - 1)
    # ind_rb = (torch.mul(box_indy2, width) + box_indx2).squeeze(2).to(torch.int64).clamp(0, height*width - 1)

    # id_feature = _tranpose_and_gather_feat(id_feature, inds)
    # 方案一：相加
    # id_feature = 0.6 * _tranpose_and_gather_feat(id_feature, inds) \
    #              + 0.1 * _tranpose_and_gather_feat(id_feature, ind_lt) \
    #              + 0.1 * _tranpose_and_gather_feat(id_feature, ind_rt) \
    #              + 0.1 * _tranpose_and_gather_feat(id_feature, ind_lb) \
    #              + 0.1 * _tranpose_and_gather_feat(id_feature, ind_rb)
    # id_feature = 0.2 * _tranpose_and_gather_feat(id_feature, inds) \
    #              + 0.2 * _tranpose_and_gather_feat(id_feature, ind_lt) \
    #              + 0.2 * _tranpose_and_gather_feat(id_feature, ind_rt) \
    #              + 0.2 * _tranpose_and_gather_feat(id_feature, ind_lb) \
    #              + 0.2 * _tranpose_and_gather_feat(id_feature, ind_rb)
    # id_feature = 0.8 * _tranpose_and_gather_feat(id_feature, inds) \
    #              + 0.1 * _tranpose_and_gather_feat(id_feature, ind_lt) \
    #              + 0.1 * _tranpose_and_gather_feat(id_feature, ind_rb)
    # id_feature = 0.4 * _tranpose_and_gather_feat(id_feature, inds) \
    #              + 0.3 * _tranpose_and_gather_feat(id_feature, ind_lt) \
    #              + 0.3 * _tranpose_and_gather_feat(id_feature, ind_rb)
    # id_feature = 0.9 * _tranpose_and_gather_feat(id_feature, inds) \
    #              + 0.05 * _tranpose_and_gather_feat(id_feature, ind_lt) \
    #              + 0.05 * _tranpose_and_gather_feat(id_feature, ind_rb)
    # id_feature = 0.98 * _tranpose_and_gather_feat(id_feature, inds) \
    #              + 0.01 * _tranpose_and_gather_feat(id_feature, ind_lt) \
    #              + 0.01 * _tranpose_and_gather_feat(id_feature, ind_rb)
    # id_feature = 0.99 * _tranpose_and_gather_feat(id_feature, inds) \
    #              + 0.005 * _tranpose_and_gather_feat(id_feature, ind_lt) \
    #              + 0.005 * _tranpose_and_gather_feat(id_feature, ind_rb)
    # id_feature = _tranpose_and_gather_feat(id_feature, inds) \
    #              + 0.1 * _tranpose_and_gather_feat(id_feature, ind_lt) \
    #              + 0.1 * _tranpose_and_gather_feat(id_feature, ind_rb)
    # id_feature = 0.95 * _tranpose_and_gather_feat(id_feature, inds) \
    #              + 0.025 * _tranpose_and_gather_feat(id_feature, ind_lt) \
    #              + 0.025 * _tranpose_and_gather_feat(id_feature, ind_rb)
    # id_feature = 0.9 * _tranpose_and_gather_feat(id_feature, inds) \
    #              + 0.025 * _tranpose_and_gather_feat(id_feature, ind_lt) \
    #              + 0.025 * _tranpose_and_gather_feat(id_feature, ind_rt) \
    #              + 0.025 * _tranpose_and_gather_feat(id_feature, ind_lb) \
    #              + 0.025 * _tranpose_and_gather_feat(id_feature, ind_rb)
    # 方案二：点乘
    # id_feature = 4 * torch.mul(_tranpose_and_gather_feat(id_feature, inds),
    #                        torch.mul(_tranpose_and_gather_feat(id_feature, ind_lt),
    #                                  _tranpose_and_gather_feat(id_feature, ind_rb)))
    # 方案三：拼接
    # id_feature = torch.cat([_tranpose_and_gather_feat(id_feature, inds), _tranpose_and_gather_feat(id_feature, ind_lt),
    #                         _tranpose_and_gather_feat(id_feature, ind_rb)], dim=2)

    return detections, inds
