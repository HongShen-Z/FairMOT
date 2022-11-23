from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import math
import logging
import numpy as np
from os.path import join

import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo

from ..common import Conv

from dcn_v2 import DCN

BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)


def get_model_url(data='imagenet', name='dla34', hash='ba72cf86'):
    return join('http://dl.yf.io/dla/models', data, '{}-{}.pth'.format(name, hash))


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class SEBlock(nn.Module):
    """ Squeeze-and-excitation block """

    def __init__(self, channels, r=16):
        super(SEBlock, self).__init__()
        self.r = r
        self.squeeze = nn.Sequential(nn.Linear(channels, channels // self.r),
                                     nn.ReLU(),
                                     nn.Linear(channels // self.r, channels),
                                     nn.Sigmoid())

    def forward(self, x):
        B, C, H, W = x.size()
        squeeze = self.squeeze(torch.mean(x, dim=(2, 3))).view(B, C, 1, 1)
        return torch.mul(x, squeeze)


class SABlock(nn.Module):
    """ Spatial self-attention block """

    def __init__(self, in_channels, out_channels):
        super(SABlock, self).__init__()
        self.attention = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
                                       nn.Sigmoid())
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)

    def forward(self, x):
        attention_mask = self.attention(x)
        features = self.conv(x)
        return torch.mul(features, attention_mask)


class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3,
                               stride=stride, padding=dilation,
                               bias=False, dilation=dilation)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=dilation,
                               bias=False, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.stride = stride
        self.downsample = downsample

    def forward(self, x, residual=None):
        if residual is None:
            residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, dilation=1):
        super(Bottleneck, self).__init__()
        expansion = Bottleneck.expansion
        bottle_planes = planes // expansion
        self.conv1 = nn.Conv2d(inplanes, bottle_planes,
                               kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(bottle_planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(bottle_planes, bottle_planes, kernel_size=3,
                               stride=stride, padding=dilation,
                               bias=False, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(bottle_planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(bottle_planes, planes,
                               kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride

    def forward(self, x, residual=None):
        if residual is None:
            residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += residual
        out = self.relu(out)

        return out


class BottleneckX(nn.Module):
    expansion = 2
    cardinality = 32

    def __init__(self, inplanes, planes, stride=1, dilation=1):
        super(BottleneckX, self).__init__()
        cardinality = BottleneckX.cardinality
        # dim = int(math.floor(planes * (BottleneckV5.expansion / 64.0)))
        # bottle_planes = dim * cardinality
        bottle_planes = planes * cardinality // 32
        self.conv1 = nn.Conv2d(inplanes, bottle_planes,
                               kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(bottle_planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(bottle_planes, bottle_planes, kernel_size=3,
                               stride=stride, padding=dilation, bias=False,
                               dilation=dilation, groups=cardinality)
        self.bn2 = nn.BatchNorm2d(bottle_planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(bottle_planes, planes,
                               kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride

    def forward(self, x, residual=None):
        if residual is None:
            residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += residual
        out = self.relu(out)

        return out


class Root(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, residual):
        super(Root, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, 1,
            stride=1, bias=False, padding=(kernel_size - 1) // 2)
        self.bn = nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.residual = residual

    def forward(self, *x):
        children = x
        x = self.conv(torch.cat(x, 1))
        x = self.bn(x)
        if self.residual:
            x += children[0]
        x = self.relu(x)

        return x


class Tree(nn.Module):
    def __init__(self, levels, block, in_channels, out_channels, stride=1,
                 level_root=False, root_dim=0, root_kernel_size=1,
                 dilation=1, root_residual=False):
        super(Tree, self).__init__()
        if root_dim == 0:
            root_dim = 2 * out_channels
        if level_root:
            root_dim += in_channels
        if levels == 1:
            self.tree1 = block(in_channels, out_channels, stride,
                               dilation=dilation)
            self.tree2 = block(out_channels, out_channels, 1,
                               dilation=dilation)
        else:
            self.tree1 = Tree(levels - 1, block, in_channels, out_channels,
                              stride, root_dim=0,
                              root_kernel_size=root_kernel_size,
                              dilation=dilation, root_residual=root_residual)
            self.tree2 = Tree(levels - 1, block, out_channels, out_channels,
                              root_dim=root_dim + out_channels,
                              root_kernel_size=root_kernel_size,
                              dilation=dilation, root_residual=root_residual)
        if levels == 1:
            self.root = Root(root_dim, out_channels, root_kernel_size,
                             root_residual)
        self.level_root = level_root
        self.root_dim = root_dim
        self.downsample = None
        self.project = None
        self.levels = levels
        if stride > 1:
            self.downsample = nn.MaxPool2d(stride, stride=stride)
        if in_channels != out_channels:
            self.project = nn.Sequential(
                nn.Conv2d(in_channels, out_channels,
                          kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM)
            )

    def forward(self, x, residual=None, children=None):
        children = [] if children is None else children
        bottom = self.downsample(x) if self.downsample else x
        residual = self.project(bottom) if self.project else bottom
        if self.level_root:
            children.append(bottom)
        x1 = self.tree1(x, residual)
        if self.levels == 1:
            x2 = self.tree2(x1)
            x = self.root(x2, x1, *children)
        else:
            children.append(x1)
            x = self.tree2(x1, children=children)
        return x


class DLA(nn.Module):
    def __init__(self, levels, channels, num_classes=1000,
                 block=BasicBlock, residual_root=False, linear_root=False):
        super(DLA, self).__init__()
        self.channels = channels
        self.num_classes = num_classes
        self.base_layer = nn.Sequential(
            nn.Conv2d(3, channels[0], kernel_size=7, stride=1,
                      padding=3, bias=False),
            nn.BatchNorm2d(channels[0], momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True))
        self.level0 = self._make_conv_level(
            channels[0], channels[0], levels[0])
        self.level1 = self._make_conv_level(
            channels[0], channels[1], levels[1], stride=2)
        self.level2 = Tree(levels[2], block, channels[1], channels[2], 2,
                           level_root=False,
                           root_residual=residual_root)
        self.level3 = Tree(levels[3], block, channels[2], channels[3], 2,
                           level_root=True, root_residual=residual_root)
        self.level4 = Tree(levels[4], block, channels[3], channels[4], 2,
                           level_root=True, root_residual=residual_root)
        self.level5 = Tree(levels[5], block, channels[4], channels[5], 2,
                           level_root=True, root_residual=residual_root)

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.weight.data.normal_(0, math.sqrt(2. / n))
        #     elif isinstance(m, nn.BatchNorm2d):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()

    def _make_level(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes:
            downsample = nn.Sequential(
                nn.MaxPool2d(stride, stride=stride),
                nn.Conv2d(inplanes, planes,
                          kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(planes, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample=downsample))
        for i in range(1, blocks):
            layers.append(block(inplanes, planes))

        return nn.Sequential(*layers)

    def _make_conv_level(self, inplanes, planes, convs, stride=1, dilation=1):
        modules = []
        for i in range(convs):
            modules.extend([
                nn.Conv2d(inplanes, planes, kernel_size=3,
                          stride=stride if i == 0 else 1,
                          padding=dilation, bias=False, dilation=dilation),
                nn.BatchNorm2d(planes, momentum=BN_MOMENTUM),
                nn.ReLU(inplace=True)])
            inplanes = planes
        return nn.Sequential(*modules)

    def forward(self, x):
        y = []
        x = self.base_layer(x)
        for i in range(6):
            x = getattr(self, 'level{}'.format(i))(x)
            y.append(x)
        return y

    def load_pretrained_model(self, data='imagenet', name='dla34', hash='ba72cf86'):
        # fc = self.fc
        if name.endswith('.pth'):
            model_weights = torch.load(data + name)
        else:
            model_url = get_model_url(data, name, hash)
            model_weights = model_zoo.load_url(model_url)
        num_classes = len(model_weights[list(model_weights.keys())[-1]])
        self.fc = nn.Conv2d(
            self.channels[-1], num_classes,
            kernel_size=1, stride=1, padding=0, bias=True)
        self.load_state_dict(model_weights)
        # self.fc = fc


def dla34(pretrained=True, **kwargs):  # DLA-34
    model = DLA([1, 1, 1, 2, 2, 1],
                [16, 32, 64, 128, 256, 512],
                block=BasicBlock, **kwargs)
    if pretrained:
        model.load_pretrained_model(data='imagenet', name='dla34', hash='ba72cf86')
    return model


class Identity(nn.Module):

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


def fill_fc_weights(layers):
    for m in layers.modules():
        if isinstance(m, nn.Conv2d):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    # 判断是否为conv2d，使用相应的初始化方式
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    # 是否为批归一化层
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


def fill_up_weights(up):
    w = up.weight.data
    f = math.ceil(w.size(2) / 2)
    c = (2 * f - 1 - f % 2) / (2. * f)
    for i in range(w.size(2)):
        for j in range(w.size(3)):
            w[0, 0, i, j] = \
                (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))
    for c in range(1, w.size(0)):
        w[c, 0, :, :] = w[0, 0, :, :]


class DeformConv(nn.Module):
    def __init__(self, chi, cho):
        super(DeformConv, self).__init__()
        self.actf = nn.Sequential(
            nn.BatchNorm2d(cho, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True)
        )
        self.conv = DCN(chi, cho, kernel_size=(3, 3), stride=1, padding=1, dilation=1, deformable_groups=1)

    def forward(self, x):
        x = self.conv(x)
        x = self.actf(x)
        return x


class IDAUp(nn.Module):

    def __init__(self, o, channels, up_f):
        super(IDAUp, self).__init__()
        for i in range(1, len(channels)):
            c = channels[i]
            f = int(up_f[i])
            proj = DeformConv(c, o)
            node = DeformConv(o, o)

            up = nn.ConvTranspose2d(o, o, f * 2, stride=f,
                                    padding=f // 2, output_padding=0,
                                    groups=o, bias=False)
            fill_up_weights(up)

            setattr(self, 'proj_' + str(i), proj)
            setattr(self, 'up_' + str(i), up)
            setattr(self, 'node_' + str(i), node)

    def forward(self, layers, startp, endp):
        for i in range(startp + 1, endp):
            upsample = getattr(self, 'up_' + str(i - startp))
            project = getattr(self, 'proj_' + str(i - startp))
            layers[i] = upsample(project(layers[i]))
            node = getattr(self, 'node_' + str(i - startp))
            layers[i] = node(layers[i] + layers[i - 1])


class DLAUp(nn.Module):
    def __init__(self, startp, channels, scales, in_channels=None):
        super(DLAUp, self).__init__()
        self.startp = startp
        if in_channels is None:
            in_channels = channels
        self.channels = channels
        channels = list(channels)
        scales = np.array(scales, dtype=int)
        for i in range(len(channels) - 1):
            j = -i - 2
            setattr(self, 'ida_{}'.format(i),
                    IDAUp(channels[j], in_channels[j:], scales[j:] // scales[j]))
            scales[j + 1:] = scales[j]
            in_channels[j + 1:] = [channels[j] for _ in channels[j + 1:]]

    def forward(self, layers):
        out = [layers[-1]]  # start with 1/32
        for i in range(len(layers) - self.startp - 1):
            ida = getattr(self, 'ida_{}'.format(i))
            ida(layers, len(layers) - i - 2, len(layers))
            out.insert(0, layers[-1])
        return out


class ReidUp(nn.Module):
    def __init__(self, in_channel, out_channel, up_radio):
        super(ReidUp, self).__init__()
        self.proj = DeformConv(in_channel, out_channel)
        self.up = nn.ConvTranspose2d(out_channel, out_channel, up_radio * 2, stride=up_radio, padding=up_radio // 2,
                                     output_padding=0,
                                     groups=out_channel, bias=False)
        self.Conv = nn.Conv2d(out_channel, 1, kernel_size=1, stride=1, bias=False)
        # self.actf = nn.Sequential(
        #     nn.BatchNorm2d(cho, momentum=BN_MOMENTUM),
        #     nn.ReLU(inplace=True)
        # )
        self.att = nn.Sigmoid()

        fill_up_weights(self.up)

    def forward(self, x):
        x_up = self.Conv(self.up(self.proj(x)))
        return self.att(x_up)


class InitialTaskPredictionModule(nn.Module):
    """ Module to make the inital task predictions """

    def __init__(self, heads, auxilary_tasks, input_channels, task_channels):
        super(InitialTaskPredictionModule, self).__init__()
        self.auxilary_tasks = auxilary_tasks

        # Per task feature refinement + decoding
        if input_channels == task_channels:
            channels = input_channels
            self.refinement = nn.ModuleDict(
                {task: nn.Sequential(BasicBlock(channels, channels), BasicBlock(channels, channels))
                 for task in self.auxilary_tasks})

        else:
            refinement = {}
            for t in auxilary_tasks:
                downsample = nn.Sequential(nn.Conv2d(input_channels, task_channels, 1, bias=False),
                                           nn.BatchNorm2d(task_channels))
                refinement[t] = nn.Sequential(BasicBlock(input_channels, task_channels, downsample=downsample),
                                              BasicBlock(task_channels, task_channels))
            self.refinement = nn.ModuleDict(refinement)

        self.decoders = nn.ModuleDict(
            {task: nn.Conv2d(task_channels, heads[task], 1) for task in self.auxilary_tasks})

    def forward(self, features_curr_scale, features_prev_scale=None):
        if features_prev_scale is not None:  # Concat features that were propagated from previous scale
            x = {t: torch.cat(
                (features_curr_scale, F.interpolate(features_prev_scale[t], scale_factor=2, mode='bilinear')), 1) for t
                in self.auxilary_tasks}

        else:
            x = {t: features_curr_scale for t in self.auxilary_tasks}

        # Refinement + Decoding
        out = {}
        for t in self.auxilary_tasks:
            out['features_%s' % t] = self.refinement[t](x[t])
            out[t] = self.decoders[t](out['features_%s' % t])

        return out


class FPM(nn.Module):
    """ Feature Propagation Module """

    def __init__(self, auxilary_tasks, per_task_channels):
        super(FPM, self).__init__()
        # General
        self.auxilary_tasks = auxilary_tasks
        self.N = len(self.auxilary_tasks)
        self.per_task_channels = per_task_channels
        self.shared_channels = int(self.N * per_task_channels)

        # Non-linear function f
        downsample = nn.Sequential(nn.Conv2d(self.shared_channels, self.shared_channels // 4, 1, bias=False),
                                   nn.BatchNorm2d(self.shared_channels // 4))
        self.non_linear = nn.Sequential(
            BasicBlock(self.shared_channels, self.shared_channels // 4, downsample=downsample),
            BasicBlock(self.shared_channels // 4, self.shared_channels // 4),
            nn.Conv2d(self.shared_channels // 4, self.shared_channels, 1))

        # Dimensionality reduction
        downsample = nn.Sequential(nn.Conv2d(self.shared_channels, self.per_task_channels, 1, bias=False),
                                   nn.BatchNorm2d(self.per_task_channels))
        self.dimensionality_reduction = BasicBlock(self.shared_channels, self.per_task_channels,
                                                   downsample=downsample)

        # SEBlock
        self.se = nn.ModuleDict({task: SEBlock(self.per_task_channels) for task in self.auxilary_tasks})

    def forward(self, x):
        # Get shared representation
        concat = torch.cat([x['features_%s' % task] for task in self.auxilary_tasks], 1)
        B, C, H, W = concat.size()
        shared = self.non_linear(concat)
        mask = F.softmax(shared.view(B, C // self.N, self.N, H, W), dim=2)  # Per task attention mask
        # mask = F.softmax(concat.view(B, C // self.N, self.N, H, W), dim=2)  # Per task attention mask
        shared = torch.mul(mask, concat.view(B, C // self.N, self.N, H, W)).view(B, -1, H, W)

        # Perform dimensionality reduction
        shared = self.dimensionality_reduction(shared)

        # Per task squeeze-and-excitation
        out = {}
        for task in self.auxilary_tasks:
            out[task] = self.se[task](shared) + x['features_%s' % task]

        return out


class MultiTaskDistillationModule(nn.Module):
    """
        Perform Multi-Task Distillation
        We apply an attention mask to features from other tasks and
        add the result as a residual.
    """

    def __init__(self, tasks, auxilary_tasks, channels):
        super(MultiTaskDistillationModule, self).__init__()
        self.tasks = tasks
        self.auxilary_tasks = auxilary_tasks
        # self.self_attention = {}
        self.proj = {}
        self.meta_tasks = {'det': self.tasks - {'id'}, 'id': 'id'}

        for t in self.tasks:
            self.proj[t] = MTAttention(k_size=3, ch=channels, s_state=True, c_state=False)
        self.proj = nn.ModuleDict(self.proj)
        self.node = nn.Sequential(MTAttention(k_size=3, ch=channels*3, s_state=False, c_state=True),
                                  Conv(channels*3, channels, k=3))
        self.c_att = MTAttention(k_size=3, ch=channels, s_state=False, c_state=True)

        # for t in self.tasks:
        #     other_tasks = [a for a in self.auxilary_tasks if a != t]
        #     self.self_attention[t] = nn.ModuleDict({a: SABlock(channels, channels) for a in other_tasks})
        # self.self_attention = nn.ModuleDict(self.self_attention)

    def forward(self, x):
        for t in self.tasks:
            x['features_%s' % t] = self.proj[t](x['features_%s' % t])
        adapters = {'id': self.node(torch.cat([x['features_%s' % t] for t in self.meta_tasks['det']], 1)),
                    'det': self.c_att(x['features_id'])}
        out = {'id': self.c_att(x['features_id']) + adapters['id']}
        for t in self.meta_tasks['det']:
            out[t] = x['features_%s' % t] + adapters['det']

        # adapters = {t: {a: self.self_attention[t][a](x['features_%s' % a])
        #                 for a in self.auxilary_tasks if a != t} for t in self.tasks}
        # out = {t: x['features_%s' % t] + torch.sum(torch.stack([v for v in adapters[t].values()]), dim=0)
        #        for t in self.tasks}
        return out


class MTAttention(nn.Module):
    """
    Multi-task attention
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """

    def __init__(self, k_size=3, ch=256, s_state=False, c_state=False):
        super(MTAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.sigmoid = nn.Sigmoid()
        # self.conv1 = Conv(ch, ch,k=1)

        self.s_state = s_state
        self.c_state = c_state

        if c_state:
            self.c_attention = nn.Sequential(nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False),
                                             nn.LayerNorm([1, ch]),
                                             nn.LeakyReLU(0.3, inplace=True),
                                             nn.Linear(ch, ch, bias=False))

        if s_state:
            self.conv_s = nn.Sequential(Conv(ch, ch // 4, k=1))
            self.s_attention = nn.Conv2d(2, 1, 7, padding=3, bias=False)

    def forward(self, x):
        # x: input features with shape [b, c, h, w]
        # b, c, h, w = x.size()

        # channel_attention
        if self.c_state:
            y_avg = self.avg_pool(x)
            y_max = self.max_pool(x)
            y_c = self.c_attention(y_avg.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1) + \
                  self.c_attention(y_max.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
            y_c = self.sigmoid(y_c)

        # spatial_attention
        if self.s_state:
            x_s = self.conv_s(x)
            avg_out = torch.mean(x_s, dim=1, keepdim=True)
            max_out, _ = torch.max(x_s, dim=1, keepdim=True)
            y_s = torch.cat([avg_out, max_out], dim=1)
            y_s = self.sigmoid(self.s_attention(y_s))

        if self.c_state and self.s_state:
            y = x * y_s * y_c + x
        elif self.c_state:
            y = x * y_c + x
        elif self.s_state:
            y = x * y_s + x
        else:
            y = x
        return y


class MTINet(nn.Module):
    """
        MTI-Net implementation based on HRNet backbone
        https://arxiv.org/pdf/2001.06902.pdf
    """

    def __init__(self, heads, backbone_channels, heads_net):
        super(MTINet, self).__init__()
        # General
        self.tasks = heads.keys()
        self.auxilary_tasks = heads.keys()
        self.num_scales = len(backbone_channels)
        self.channels = backbone_channels

        # Feature Propagation Module
        # self.fpm_scale_3 = FPM(self.auxilary_tasks, self.channels[3])
        # self.fpm_scale_2 = FPM(self.auxilary_tasks, self.channels[2])
        # self.fpm_scale_1 = FPM(self.auxilary_tasks, self.channels[1])

        # Initial task predictions at multiple scales
        # self.scale_0 = InitialTaskPredictionModule(
        #     heads, self.auxilary_tasks, self.channels[0] + self.channels[1], self.channels[0])
        # self.scale_1 = InitialTaskPredictionModule(
        #     heads, self.auxilary_tasks, self.channels[1] + self.channels[2], self.channels[1])
        # self.scale_2 = InitialTaskPredictionModule(
        #     heads, self.auxilary_tasks, self.channels[2] + self.channels[3], self.channels[2])
        # self.scale_3 = InitialTaskPredictionModule(
        #     heads, self.auxilary_tasks, self.channels[3], self.channels[3])

        self.scale_0 = InitialTaskPredictionModule(
            heads, self.auxilary_tasks, self.channels[0], self.channels[0])
        self.fpm_scale_0 = FPM(self.auxilary_tasks, self.channels[0])

        # Distillation at multiple scales
        # self.distillation_scale_0 = MultiTaskDistillationModule(self.tasks, self.auxilary_tasks, self.channels[0])
        # self.distillation_scale_1 = MultiTaskDistillationModule(self.tasks, self.auxilary_tasks, self.channels[1])
        # self.distillation_scale_2 = MultiTaskDistillationModule(self.tasks, self.auxilary_tasks, self.channels[2])
        # self.distillation_scale_3 = MultiTaskDistillationModule(self.tasks, self.auxilary_tasks, self.channels[3])

        # Feature aggregation through HRNet heads
        self.heads_net = heads_net

    def forward(self, x):
        # img_size = x.size()[-2:]
        out = {}

        # Backbone
        # x = self.backbone(x)

        # Predictions at multiple scales
        # # Scale 3
        # x_3 = self.scale_3(x[3])
        # x_3_fpm = self.fpm_scale_3(x_3)
        # # Scale 2
        # x_2 = self.scale_2(x[2], x_3_fpm)
        # x_2_fpm = self.fpm_scale_2(x_2)
        # # Scale 1
        # x_1 = self.scale_1(x[1], x_2_fpm)
        # x_1_fpm = self.fpm_scale_1(x_1)
        # # Scale 0
        # x_0 = self.scale_0(x[0], x_1_fpm)
        #
        # out['deep_supervision'] = {'scale_0': x_0, 'scale_1': x_1, 'scale_2': x_2, 'scale_3': x_3}

        x_0 = self.scale_0(x)
        out['deep_supervision'] = {'scale_0': x_0}
        features_0 = self.fpm_scale_0(x_0)

        # Distillation + Output
        # features_0 = self.distillation_scale_0(x_0)
        # features_1 = self.distillation_scale_1(x_1)
        # features_2 = self.distillation_scale_2(x_2)
        # features_3 = self.distillation_scale_3(x_3)
        # multi_scale_features = {t: [features_0[t], features_1[t], features_2[t], features_3[t]] for t in self.tasks}

        # Feature aggregation
        for t in self.tasks:
            # out[t] = F.interpolate(self.heads_net[t](multi_scale_features[t]), img_size, mode='bilinear')
            # out[t] = self.heads_net[t](multi_scale_features[t])
            out[t] = self.heads_net[t](features_0[t])

        return out


class CenterHead(nn.Module):
    def __init__(self, backbone_channels, heads, head, head_conv=256):
        super(CenterHead, self).__init__()
        if head_conv > 0:
            self.fc = nn.Sequential(
                nn.Conv2d(backbone_channels[0], head_conv, kernel_size=3, padding=1, bias=True),
                nn.BatchNorm2d(head_conv, momentum=0.1),
                nn.ReLU(inplace=True),
                nn.Conv2d(head_conv, heads[head], kernel_size=1, stride=1, padding=0, bias=True))
            if 'hm' in head:
                self.fc[-1].bias.data.fill_(-2.19)
            else:
                fill_fc_weights(self.fc)
        else:
            self.fc = nn.Conv2d(backbone_channels[0], heads[head],
                                kernel_size=1, stride=1, padding=0, bias=True)
            if 'hm' in head:
                self.fc.bias.data.fill_(-2.19)
            else:
                fill_fc_weights(self.fc)

    def forward(self, x):
        # x0_h, x0_w = x[0].size(2), x[0].size(3)
        # x1 = F.interpolate(x[1], (x0_h, x0_w), mode='bilinear')
        # x2 = F.interpolate(x[2], (x0_h, x0_w), mode='bilinear')
        # x3 = F.interpolate(x[3], (x0_h, x0_w), mode='bilinear')
        #
        # x = torch.cat([x[0], x1, x2, x3], 1)
        x = self.fc(x)
        return x


class HighResolutionHead(nn.Module):
    def __init__(self, backbone_channels, num_outputs):
        super(HighResolutionHead, self).__init__()
        last_inp_channels = sum(backbone_channels)
        self.last_layer = nn.Sequential(
            nn.Conv2d(
                in_channels=last_inp_channels,
                out_channels=256,
                kernel_size=1,
                stride=1,
                padding=0),
            nn.BatchNorm2d(256, momentum=0.1),
            nn.ReLU(inplace=False),
            nn.Conv2d(
                in_channels=256,
                out_channels=num_outputs,
                kernel_size=1,
                stride=1,
                padding=0))

    def forward(self, x):
        x0_h, x0_w = x[0].size(2), x[0].size(3)
        x1 = F.interpolate(x[1], (x0_h, x0_w), mode='bilinear')
        x2 = F.interpolate(x[2], (x0_h, x0_w), mode='bilinear')
        x3 = F.interpolate(x[3], (x0_h, x0_w), mode='bilinear')

        x = torch.cat([x[0], x1, x2, x3], 1)
        x = self.last_layer(x)
        return x


class DLASeg(nn.Module):
    def __init__(self, base_name, heads, pretrained, down_ratio, final_kernel,
                 last_level, head_conv, out_channel=0):
        super(DLASeg, self).__init__()
        assert down_ratio in [2, 4, 8, 16]
        self.first_level = int(np.log2(down_ratio))
        self.last_level = last_level
        self.base = globals()[base_name](pretrained=pretrained)
        channels = self.base.channels
        scales = [2 ** i for i in range(len(channels[self.first_level:]))]  # [1, 2, 4, 8]
        self.dla_up = DLAUp(self.first_level, channels[self.first_level:], scales)

        if out_channel == 0:
            out_channel = channels[self.first_level]

        self.ida_up = IDAUp(out_channel, channels[self.first_level:self.last_level],
                            [2 ** i for i in range(self.last_level - self.first_level)])

        # heads_net = nn.ModuleDict(
        #     {head: HighResolutionHead(channels[self.first_level:], heads[head]) for head in heads})
        heads_net = nn.ModuleDict(
            {head: CenterHead(channels[self.first_level:], heads, head, head_conv) for head in heads})
        self.mti_net = MTINet(heads, channels[self.first_level:], heads_net)
        self.mti_net.apply(weight_init)

        # self.RA_3 = ReidUp(channels[-1], channels[-2], 2)
        # self.RA_2 = ReidUp(channels[-2], channels[-3], 2)
        # self.RA_1 = ReidUp(channels[-3], channels[-4], 2)
        # self.DA_3 = ReidUp(channels[-1], channels[-2], 2)
        # self.DA_2 = ReidUp(channels[-2], channels[-3], 2)
        # self.DA_1 = ReidUp(channels[-3], channels[-4], 2)

        # self.heads = heads
        # # self.det_heads = dict([(key, heads[key]) for key in ['hm', 'wh', 'reg']])
        # # self.reid_heads = dict([('id', heads['id'])])
        # for head in self.heads:
        #     classes = self.heads[head]
        #     if head_conv > 0:
        #         fc = nn.Sequential(
        #             nn.Conv2d(channels[self.first_level], head_conv,
        #                       kernel_size=3, padding=1, bias=True),
        #             nn.ReLU(inplace=True),
        #             nn.Conv2d(head_conv, classes,
        #                       kernel_size=final_kernel, stride=1,
        #                       padding=final_kernel // 2, bias=True))
        #         if 'hm' in head:
        #             fc[-1].bias.data.fill_(-2.19)
        #         else:
        #             fill_fc_weights(fc)
        #     else:
        #         fc = nn.Conv2d(channels[self.first_level], classes,
        #                        kernel_size=final_kernel, stride=1,
        #                        padding=final_kernel // 2, bias=True)
        #         if 'hm' in head:
        #             fc.bias.data.fill_(-2.19)
        #         else:
        #             fill_fc_weights(fc)
        #     self.__setattr__(head, fc)

    def forward(self, x):
        x = self.base(x)
        x = self.dla_up(x)
        # x[0] (1,64,152,272)
        # x[1] (1,128,76,136)
        # x[2] (1,256,38,68)
        # x[3] (1,512,19,34)

        D = []
        for i in range(self.last_level - self.first_level + 1):
            D.append(x[i].clone())
        self.ida_up(D, 0, len(D))

        # det_att = self.DA_3(x[3])
        # det_att = self.DA_2(x[2] * det_att)
        # det_att = self.DA_1(x[1] * det_att)
        # D = x[0] * det_att

        # reid_att = self.RA_3(x[3])
        # reid_att = self.RA_2(x[2] * reid_att)
        # reid_att = self.RA_1(x[1] * reid_att)
        # R = x[0] * reid_att

        # z = {}
        # for head in self.heads:
        #     z[head] = self.__getattr__(head)(D[-1])
        #     z[head] = self.__getattr__(head)(D)

        # for head in self.reid_heads:
        #     z[head] = self.__getattr__(head)(R)

        out = self.mti_net(D[-1])
        return out


def get_pose_net(num_layers, heads, head_conv=256, down_ratio=4):
    model = DLASeg('dla{}'.format(num_layers), heads,
                   pretrained=False,
                   down_ratio=down_ratio,
                   final_kernel=1,
                   last_level=5,
                   head_conv=head_conv)
    return model
