# -*- coding:utf-8 -*-
"""*****************************************************************************
Authors: Yu wenlong
Description:
Functions:
*************************************Import***********************************"""
import torch
import torch.nn.functional as F
from torch import nn
import math
import numpy as np
from .backbone_bases import build_backbone
from util.misc import compu_featpart, cate_num_all_dataset, num_coarse_cate_dataset

from .criterion import SetCriterion


class CFDG(nn.Module):
    def __init__(self, backbone_f, backbone_c, args):

        super(CFDG, self).__init__()
        self.cate_num_all = cate_num_all_dataset[args.dataset]
        self.num_coarse_cate = num_coarse_cate_dataset[args.dataset]
        self.feat_dim = args.feat_dim  # 256
        self.feat_num = args.feat_num  # 128
        self.feat_len = 2048
        self.device = args.device
        self.batch_size = args.batch_size["train"]
        self.max_iter = torch.tensor(args.max_iteration, dtype=torch.float64).requires_grad_(False)
        self.cfdg = args.bn_cfdg
        self.cnfd_type = args.cnfd_type
        self.b_bkb_poor = False

        self.b_avgpool = True
        self.conv_type = 1

        self.model_type = 'para_conv'
        self.b_bkb_c = args.b_bkb_c
        self.backbone_f = backbone_f

        if args.b_bkb_c is True:  # Double backbone
            self.backbone_c = backbone_c

        # Check
        assert 'conv' in self.model_type or 'mlp' in self.model_type, 'illegal args.model_type'

        self.feat_len = 2048

        feat_ratio, feat_part = compu_featpart(args)
        self.feat_ratio = feat_ratio
        self.feat_part = feat_part

        self.feat_caus = self.feat_part[-1]
        self.nfeat_caus = self.feat_dim
        self.nfeat_cnfd = 0

        if 'conv' in self.model_type:
            btnk_Layer = BottleNeck_Layer_conv1
            if self.b_avgpool is True:
                self.pred_len = self.nfeat_caus

        if self.cfdg is True:
            if 'conv' in self.model_type:
                self.c3_bottleneck_layer = btnk_Layer(self.feat_len, self.feat_dim)
                if 'para' in self.model_type:
                    self.c2_bottleneck_layer = btnk_Layer(self.feat_len, self.feat_dim)
                    self.c1_bottleneck_layer = btnk_Layer(self.feat_len, self.feat_dim)
                    self.f0_bottleneck_layer = btnk_Layer(self.feat_len, self.feat_dim)

                self.c3_classifier_layer = predictor(self.pred_len, self.cate_num_all[3])
                self.c2_classifier_layer = predictor(self.pred_len, self.cate_num_all[2])
                self.c1_classifier_layer = predictor(self.pred_len, self.cate_num_all[1])
                self.f0_classifier_layer = predictor(self.pred_len, self.cate_num_all[0])

        if 'conv' in self.model_type and self.b_avgpool is True:
            self.dis_lin = nn.AdaptiveAvgPool2d((1, 1))
        self.b_da_split = False

    #
    def initialize_weights(self, module2be_init):
        for m in module2be_init.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight, gain=1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def save_bkb_gradients(self, xs_gradients):
        self.xs_gradients = xs_gradients

    def get_bkb_gradients(self):
        return self.xs_gradients

    def forward(self, x, iter_time=1, onlylidu=''):

        B, L, _, _ = x.shape
        split = int(B/2)
        out_feat_btnk = []
        logits_coarse = []
        logits_cnfd = []

        feat_sp, feat_f = self.backbone_f(x)
        feat_f_use = feat_sp
        #
        if self.cfdg is True:
            if self.b_bkb_c is False:
                c3_feat_btnk = self.c3_bottleneck_layer(feat_f_use)
                if 'para' in self.model_type:
                    c2_feat_btnk = self.c2_bottleneck_layer(feat_f_use)
                    c1_feat_btnk = self.c1_bottleneck_layer(feat_f_use)
                    f0_feat_btnk = self.f0_bottleneck_layer(feat_f_use)

            else:  # 双路
                feat_sp_c, feat_c = self.backbone_c(x)

                if self.b_bkb_poor is True:
                    feat_c_use = feat_c
                else:

                    feat_c_use = feat_sp_c

                if 'para' in self.model_type:
                    c3_feat_btnk = self.c3_bottleneck_layer(feat_c_use)
                    c2_feat_btnk = self.c2_bottleneck_layer(feat_c_use)
                    c1_feat_btnk = self.c1_bottleneck_layer(feat_c_use)
                    f0_feat_btnk = self.f0_bottleneck_layer(feat_f_use)

            out_feat_btnk.append(feat_sp)
            f0_feat_btnk_split = f0_feat_btnk

            out_feat_btnk.append(c3_feat_btnk)
            out_feat_btnk.append(c2_feat_btnk)
            out_feat_btnk.append(c1_feat_btnk)
            out_feat_btnk.append(f0_feat_btnk_split)

            c3_feat_btnk = c3_feat_btnk[:, :self.feat_caus, ...]
            c2_feat_btnk = c2_feat_btnk[:, :self.feat_caus, ...]
            c1_feat_btnk = c1_feat_btnk[:, :self.feat_caus, ...]
            f0_feat_btnk = f0_feat_btnk[:, :self.feat_caus, ...]

            if 'conv' in self.model_type and self.b_avgpool is True:
                c3_feat_btnk = self.dis_lin(c3_feat_btnk).view(c3_feat_btnk.size(0), -1)
                c2_feat_btnk = self.dis_lin(c2_feat_btnk).view(c2_feat_btnk.size(0), -1)
                c1_feat_btnk = self.dis_lin(c1_feat_btnk).view(c1_feat_btnk.size(0), -1)
                f0_feat_btnk = self.dis_lin(f0_feat_btnk).view(f0_feat_btnk.size(0), -1)

            logits_coarse.append(self.c1_classifier_layer(c1_feat_btnk))  # [32, 122]
            logits_coarse.append(self.c2_classifier_layer(c2_feat_btnk))  # [32, 38]
            logits_coarse.append(self.c3_classifier_layer(c3_feat_btnk))  # [32, 14]
            logits_f = self.f0_classifier_layer(f0_feat_btnk)         # [32, 200]

            #
            logits_fine_soft = nn.Softmax(dim=1)(logits_f).detach()  # [72,200]

        elif self.cfdg is False:
            if self.b_bkb_c is True:
                f0_feat_btnk = self.f0_bottleneck_layer(feat_f_use)
                feat_sp_c, feat_c = self.backbone_c(x)
                if self.b_bkb_poor is True:
                    feat_c_use = feat_c
                else:
                    feat_c_use = feat_sp_c

                c_feat_btnk = self.c_bottleneck_layer(feat_c_use)

            else:
                c_feat_btnk = f0_feat_btnk = self.f0_bottleneck_layer(feat_f_use)

            if self.training is not True or self.b_da_split is not True:
                out_feat_btnk.append(feat_sp)
                out_feat_btnk.append(c_feat_btnk)
                out_feat_btnk.append(f0_feat_btnk)
            if 'conv' in self.model_type and self.b_avgpool is True:
                f0_feat_btnk = self.dis_lin(f0_feat_btnk).view(f0_feat_btnk.size(0), -1)
                c_feat_btnk = self.dis_lin(c_feat_btnk).view(c_feat_btnk.size(0), -1)

            logits_coarse = []
            logits_coarse.append(self.c1_classifier_layer(c_feat_btnk))
            logits_coarse.append(self.c2_classifier_layer(c_feat_btnk))
            logits_coarse.append(self.c3_classifier_layer(c_feat_btnk))
            logits_f = self.f0_classifier_layer(f0_feat_btnk)

        if len(out_feat_btnk[0].shape) == 4:
            out_feat_btnk[0] = out_feat_btnk[0].view(out_feat_btnk[0].size(0), out_feat_btnk[0].size(1), -1)
            if self.b_da_split is not True or self.training is not True:
                pass
            else:
                B = split
            for iii in range(1, len(out_feat_btnk)):
                if len(out_feat_btnk[iii].shape) == 4:
                    try:
                        out_feat_btnk[iii] = out_feat_btnk[iii].view(B, out_feat_btnk[iii].size(1), -1)
                    except:
                        print('over')

        return logits_f, logits_coarse, None, out_feat_btnk, None, None, None


class predictor(nn.Module):
    def __init__(self, feature_len, cate_num):
        super(predictor, self).__init__()
        self.classifier = nn.Linear(feature_len, cate_num)
        self.classifier.weight.data.normal_(0, 0.01)
        self.classifier.bias.data.fill_(0.0)

    def forward(self, features):
        activations = self.classifier(features)
        return (activations)


class BottleNeck_Layer_linear(nn.Module):
    def __init__(self, in_dim, out_dim):

        super(BottleNeck_Layer_linear, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim

        self.bottleneck = nn.Linear(in_dim, out_dim)
        self.relu = nn.ReLU()
        self.drop_out = nn.Dropout(0.5)
        self.initialize_weights()

    def initialize_weights(self):
        # nn.init.xavier_uniform_(self.conv.weight, gain=1)
        # nn.init.constant_(self.conv.bias, 0)
        self.bottleneck.weight.data.normal_(0, 0.005)
        self.bottleneck.bias.data.fill_(0.1)

    def forward(self, x):
        x = self.bottleneck(x)
        x = self.relu(x)
        x = self.drop_out(x)
        return x


class BottleNeck_Layer_conv1(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(BottleNeck_Layer_conv1, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim

        self.btnk = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size=1),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=0.5),
            )

        self.initialize_weights()

    def initialize_weights(self):
        for m in self.btnk.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight, gain=1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        identity = x
        out = self.btnk(x)
        return out


class BottleNeck_Layer_conv1i(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(BottleNeck_Layer_conv1, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim

        self.btnk = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size=1),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True),
            )

        self.initialize_weights()

    def initialize_weights(self):
        for m in self.btnk.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight, gain=1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        identity = x
        out = self.btnk(x)
        out = out + identity
        return out


def build_model(args):
    # coarse-grained feature extractor + coarse-grained label predictor
    if args.b_bkb_c is True:
        backbone_c = build_backbone(args)
    else:
        backbone_c = None

    backbone_f = build_backbone(args)

    model = CFDG(backbone_f, backbone_c, args)
    model.train(True)
    criterion = SetCriterion(args)

    return model, criterion













