# -*- coding:utf-8 -*-
"""*****************************************************************************
Time:    2023- 06- 29
Authors: Yu wenlong  and  DRAGON_501
Description:
Functions: 
Input: 
Output: 
Note:

Link: 

*************************************Import***********************************"""
import torch
import torch.nn.functional as F
from torch import nn

from util.misc import strategy_progressive, entropy_loss_func, feat_sim_cos_T1, compu_featpart, cate_num_all_dataset

criterion_type = {
    "classifier": nn.CrossEntropyLoss(),
    "kl_loss": nn.KLDivLoss(size_average=False),
    "adversarial": nn.BCELoss()
}


class SetCriterion(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.num_coarse_g = args.granu_num - 1
        self.device = args.device

        self.B = self.args.batchsize
        self.G = args.granu_num
        self.ele_g = list(range(self.G))
        self.cate_num_all = cate_num_all_dataset[args.dataset]
        self.g_dataset_lenth = len(self.cate_num_all)

        feat_ratio, feat_part = compu_featpart(args)
        self.feat_ratio = feat_ratio
        self.feat_part = feat_part

        self.acc = {}
        self.accu = {}
        self.loss = {}
        self.loss_accu = {}
        self.acc_accu = {}
        self.fine_coarse_map = args.fine_coarse_map

        self.loss['total_loss'] = torch.tensor(0.0)

        self.loss['fine_classifier_loss'] = torch.tensor(0.0)
        self.loss['coarse_classifier_loss'] = torch.tensor(0.0)
        self.loss['classifier_loss'] = torch.tensor(0.0)
        self.loss['transfer_loss'] = torch.tensor(0.0)
        self.loss['entropy_loss_source'] = torch.tensor(0.0)
        self.loss['entropy_loss_target'] = torch.tensor(0.0)
        self.loss['loss_same_sa_diff_g_com'] = torch.tensor(0.0)
        self.loss['loss_diff_sa_same_g_pvt'] = torch.tensor(0.0)
        self.loss['loss_diff_sa_diff_g_com'] = torch.tensor(0.0)

        for i_e_g in self.ele_g[::-1]:
            self.acc[f'acc_{i_e_g}'] = 0.0
            self.acc_accu[f'acc_{i_e_g}'] = torch.zeros((1), dtype=torch.float32, device='cuda')
            self.accu[f'acc_{i_e_g}'] = torch.zeros((1, self.cate_num_all[i_e_g]), dtype=torch.float32, device='cuda')

        self.accu_acc = torch.zeros((1, self.G), dtype=torch.float64, device='cuda')
        self.accu_labels = torch.zeros((1, self.g_dataset_lenth), dtype=torch.float64, device='cuda')

        self.init_loss_and_oth()

    def set_da_domain_label(self, domain_labels):
        self.domain_labels = domain_labels

    def add_param(self, fine_coarse_map):
        pass

    def accumulate_logits(self, f_logits, c_logits, labels):
        self.accu['acc_0'] = torch.cat((self.accu['acc_0'], f_logits.data.float()), 0)
        for i_e_g in self.ele_g[::-1][:-1]:
            self.accu[f'acc_{i_e_g}'] = torch.cat((self.accu[f'acc_{i_e_g}'], c_logits[i_e_g - 1].data.float()), 0)
        self.accu_labels = torch.cat((self.accu_labels, labels.data.float()), 0)

    def accumulate_loss(self, loss_dict):
        for key, value in loss_dict.items():
            self.loss_accu[key] = torch.cat((self.loss_accu[key], loss_dict[key]), 0)

    def acc_compute_accu(self):
        num_allsamples = float(self.accu_labels.size()[0] - 1)
        for i, (i_key, i_value) in enumerate(self.accu.items()):
            _, predict = torch.max(self.accu[i_key][1:], 1)
            self.acc[i_key] = torch.sum(
                torch.squeeze(predict).float() == self.accu_labels[1:, i]).item() / num_allsamples
        self.init_accu()
        return self.acc, num_allsamples

    def acc_compute(self, f_logits, c_logits, labels):
        num_allsamples = float(labels.size()[0])
        for i, (i_key, i_value) in enumerate(self.acc.items()):
            if i == 0:
                _, predict = torch.max(f_logits.data.float(), 1)
                self.acc['acc_0'] = torch.sum(torch.squeeze(predict).float() == labels[:, 0]).item() / num_allsamples
                self.acc_accu[f'acc_0'] = torch.cat((self.acc_accu[f'acc_0'],
                                                     torch.tensor(self.acc['acc_0'], dtype=torch.float32,
                                                                  device='cuda').unsqueeze(0)))
            elif c_logits is not None:
                _, predict = torch.max(c_logits[i - 1].data.float(), 1)
                self.acc[f'acc_{i}'] = torch.sum(torch.squeeze(predict).float() == labels[:, i]).item() / num_allsamples
                self.acc_accu[f'acc_{i}'] = torch.cat((self.acc_accu[f'acc_{i}'],
                                                       torch.tensor(self.acc[f'acc_{i}'], dtype=torch.float32,
                                                                    device='cuda').unsqueeze(0)))
        acc_all = torch.tensor([v for v in self.acc.values()], dtype=torch.float32, device='cuda').unsqueeze(0)
        self.accu_acc = torch.cat((self.accu_acc, acc_all), 0)
        return self.acc, num_allsamples

    def output_acc(self):
        num_allsamples = float(len(self.accu_acc) - 1)
        for i_e_g in self.ele_g:
            self.acc[f'acc_{i_e_g}'] = torch.sum(self.acc_accu[f'acc_{i_e_g}'][1:]) / num_allsamples
            self.acc_accu[f'acc_{i_e_g}'] = torch.zeros((1), dtype=torch.float32, device='cuda')

        self.accu_acc = torch.zeros((1, self.G), dtype=torch.float32, device='cuda')
        return self.acc, num_allsamples

    def output_loss(self):
        num_allsamples = float(len(self.loss_accu))
        loss_output = {}
        for key, value in self.loss_accu.items():
            loss_output[key] = torch.sum(self.loss_accu) / num_allsamples
        return loss_output, num_allsamples

    def init_loss_and_oth(self):
        for key, value in self.loss.items():
            self.loss[key] = torch.tensor(0.0).to(self.device)

    def init_accu(self):
        for i_e_g in self.ele_g[::-1]:
            self.accu[f'acc_{i_e_g}'] = torch.zeros((1, self.cate_num_all[i_e_g]), dtype=torch.float32, device='cuda')
        self.accu_labels = torch.zeros((1, self.G), dtype=torch.float32, device='cuda')

    def output_all_loss(self):
        return self.loss

    def forward(self, labels_source, logits_fine, logits_coarse, out_feat_btnk, iter_num):

        fine_labels_source_cpu = labels_source[:, 0].view(-1, 1)

        labels_source = labels_source.to(self.device)
        coarse_labels_source = []
        for i in range(self.num_coarse_g):
            coarse_labels_source.append(labels_source[:, i + 1])

        logits_fine_source = logits_fine
        fine_labels_onehot = torch.zeros(logits_fine_source.size()).scatter_(1, fine_labels_source_cpu, 1).to(self.device)
        lambda_progressive = strategy_progressive(iter_num, self.args.initial_smooth, self.args.final_smooth,
                                                  self.args.max_iteration, self.args.smooth_stratege)
        if logits_coarse is not None:
            logits_coarse_detach = []
            logits_coarse_softmax = []
            logits_coarse_detach_extended = []
            for i in range(len(logits_coarse)):
                logits_coarse_detach.append(logits_coarse[i].detach())
                logits_coarse_detach_extended.append(torch.zeros(logits_fine_source.size()))
                logits_coarse_softmax.append(nn.Softmax(dim=1)(logits_coarse[i]).detach())

            for i in range(len(logits_coarse_detach)):
                for j in range(len(self.fine_coarse_map)):
                    logits_coarse_detach_extended[i][:, j] = logits_coarse_detach[i][:, self.fine_coarse_map[j][i + 1]]

            for i in range(len(logits_coarse_detach_extended)):
                logits_coarse_detach_extended[i] = nn.Softmax(dim=1)(logits_coarse_detach_extended[i]).to(self.device)

            for i in range(1, len(logits_coarse_detach_extended)):
                logits_coarse_detach_extended[0] += logits_coarse_detach_extended[i]

            if self.args.bn_align is True:
                labels_onehot_smooth = (1 - lambda_progressive) * fine_labels_onehot + lambda_progressive * (
                logits_coarse_detach_extended[0]) / self.num_coarse_g
                fine_classifier_loss = criterion_type["kl_loss"](nn.LogSoftmax(dim=1)(logits_fine_source),
                                                                 labels_onehot_smooth)
                fine_classifier_loss = fine_classifier_loss / self.args.batch_size["train"]
            else:
                labels_onehot_smooth = (1 - lambda_progressive + 0.1) * fine_labels_onehot
                fine_classifier_loss = criterion_type["kl_loss"](nn.LogSoftmax(dim=1)(logits_fine_source),
                                                                 labels_onehot_smooth)
                fine_classifier_loss = fine_classifier_loss / self.args.batch_size["train"]

            for i in range(len(logits_coarse)):
                if i == 0:
                    coarse_classifier_loss = criterion_type["classifier"](logits_coarse[i], coarse_labels_source[i])
                else:
                    coarse_classifier_loss += criterion_type["classifier"](logits_coarse[i],
                                                                           coarse_labels_source[i])
            else:
                coarse_classifier_loss = torch.tensor(0.0).to(self.device)

        else:
            coarse_classifier_loss = torch.tensor(0.0).to(self.device)

            labels_onehot_smooth = (1 - lambda_progressive + 0.1) * fine_labels_onehot
            fine_classifier_loss = criterion_type["kl_loss"](nn.LogSoftmax(dim=1)(logits_fine_source),
                                                             labels_onehot_smooth)
            fine_classifier_loss = fine_classifier_loss / self.args.batch_size["train"]

            logits_coarse_softmax = None

        if self.args.b_f_ce is True:
            if self.args.b_lamda is True:
                fine_classifier_loss = (1 - lambda_progressive + 0.1) * criterion_type["classifier"](logits_fine, labels_source[:, 0])
            else:
                fine_classifier_loss = criterion_type["classifier"](logits_fine, labels_source[:, 0])

        classifier_loss = fine_classifier_loss + coarse_classifier_loss

        entropy_loss_source = entropy_loss_func(nn.Softmax(dim=1)(logits_fine))
        entropy_loss_target = torch.tensor(0.0)
        transfer_loss = torch.tensor(0.0)
        self.acc, _ = self.acc_compute(nn.Softmax(dim=1)(logits_fine), logits_coarse_softmax, labels_source)

        total_loss = classifier_loss + transfer_loss + \
                     entropy_loss_source * self.args.c_loss_entropy_source + \
                     entropy_loss_target * self.args.c_loss_entropy_target

        self.loss['fine_classifier_loss'] += fine_classifier_loss
        self.loss['coarse_classifier_loss'] += coarse_classifier_loss
        self.loss['classifier_loss'] += classifier_loss
        self.loss['transfer_loss'] += transfer_loss.item()
        self.loss['entropy_loss_target'] += entropy_loss_target.item()
        self.loss['entropy_loss_source'] += entropy_loss_source.item()

        if self.args.bn_cfdg is True:
            feat_part = self.feat_part

            #####################################################################################################
            if self.args.b_part_oth is True:
                same_sa_same_g_feat_dis_all_part = torch.zeros((len(out_feat_btnk)), dtype=torch.float64,
                                                               device='cuda')  # 5
                if self.args.b_part_oth is True:
                    for i_g in range(0, len(out_feat_btnk)):
                        g_feat_btnk = out_feat_btnk[i_g]

                        i_g_feat_c = torch.sum(g_feat_btnk[:, :feat_part[0], :], dim=1)
                        i_g_feat_p = torch.sum(g_feat_btnk[:, feat_part[0]:feat_part[1], :], dim=1)
                        i_g_feat_n = torch.sum(g_feat_btnk[:, feat_part[1]:, :], dim=1)
                        i_g_feat_part_proto = []
                        for i_sample in range(i_g_feat_c.shape[0]):
                            i_g_feat_part_proto.append(
                                torch.stack((i_g_feat_c[i_sample], i_g_feat_p[i_sample], i_g_feat_n[i_sample]),
                                            dim=0))
                        i_g_feat_part_proto = torch.stack(i_g_feat_part_proto)

                        I_g = torch.eye(3).cuda().repeat(i_g_feat_c.shape[0], 1, 1)
                        i_g_feat_part_proto = F.normalize(i_g_feat_part_proto, p=2, dim=2)
                        i_g_feat_part_proto_dis = torch.matmul(i_g_feat_part_proto,
                                                               i_g_feat_part_proto.permute(0, 2, 1))

                        i_g_feat_part_proto_dis = i_g_feat_part_proto_dis - I_g

                        same_sa_same_g_feat_dis_all_part[i_g] = torch.norm(i_g_feat_part_proto_dis, p=2)

                    loss_part_oth = torch.sum(same_sa_same_g_feat_dis_all_part[1:self.G + 1]) / \
                                    self.args.batch_size["train"] / self.G

                    if self.args.b_part_oth is True:
                        total_loss = total_loss + loss_part_oth * self.args.c_part_oth
                    else:
                        pass
            else:
                loss_part_oth = torch.tensor(0.0)

            ####################################################################################################
            if len(out_feat_btnk) > 1:
                out_feat_btnk = torch.stack(out_feat_btnk[1:])  # [4,10,256,49]
            if self.args.b_same_sa_diff_g_com is True:
                same_sa_diff_g_com_dis_all = torch.zeros(self.num_coarse_g, dtype=torch.float64, device='cuda')
                for i_g in range(self.num_coarse_g):
                    same_sa_diff_g_com_dis_all[i_g] = feat_sim_cos_T1(out_feat_btnk[i_g, :, :feat_part[0], :],
                                                                      out_feat_btnk[i_g + 1, :, :feat_part[0], :])
                loss_same_sa_diff_g_com = torch.sum(same_sa_diff_g_com_dis_all) / self.num_coarse_g
                if self.args.b_same_sa_diff_g_com is True:
                    total_loss = total_loss - loss_same_sa_diff_g_com * self.args.c_same_sa_diff_g_com
                else:
                    pass
            else:
                loss_same_sa_diff_g_com = torch.tensor(0.0)

            ###################################################################################################
            if self.args.b_diff_sa_same_g_pvt is True:
                diff_sa_same_g_pvt_dis_all = torch.zeros(self.G, dtype=torch.float64, device='cuda')
                all_num_cat = 0
                for i_g in range(self.G):
                    g_lables_sc = labels_source[:, self.G - 1 - i_g]

                    g_lables_sc_set = torch.unique(g_lables_sc, dim=0)
                    g_num_cat = len(g_lables_sc_set)

                    g_feat_btnk = out_feat_btnk[i_g, :, feat_part[0]:feat_part[1], :]  # [72,64,256] 获取那40%特征、
                    diff_sa_same_g_pvt_pttp_all = torch.zeros((g_num_cat, g_feat_btnk.shape[1]),
                                                              dtype=torch.float32, device='cuda')

                    for i_g_num_cat in range(g_num_cat):
                        g_i_cat = g_lables_sc_set[i_g_num_cat]
                        g_i_feat_index = (g_lables_sc == g_i_cat).nonzero(as_tuple=True)[0]
                        g_i_feat_btnk = torch.index_select(g_feat_btnk, 0, g_i_feat_index)

                        diff_sa_same_g_pvt_pttp_all[i_g_num_cat] = torch.mean(torch.mean(g_i_feat_btnk, dim=2),
                                                                              dim=0)
                    all_num_cat += g_num_cat

                    diff_sa_same_g_pvt_dis_all[i_g] = feat_sim_cos_T1(diff_sa_same_g_pvt_pttp_all,  method='diag-I')
                loss_diff_sa_same_g_pvt = torch.sum(diff_sa_same_g_pvt_dis_all) / all_num_cat

                if self.args.b_diff_sa_same_g_pvt is True:
                    total_loss = total_loss + loss_diff_sa_same_g_pvt * self.args.c_diff_sa_same_g_pvt
                else:
                    pass
            else:
                loss_diff_sa_same_g_pvt = torch.tensor(0.0)
            ##################################################################################################
            if self.args.b_diff_sa_diff_g_com is True:
                diff_sa_diff_g_dis_com_all = torch.zeros(self.G - 1, dtype=torch.float64, device='cuda')
                all_num_cat = 0
                for i_g in range(self.G - 1):
                    g_lables_father = labels_source[:, self.G - 1 - i_g]
                    g_lables_father_set = torch.unique(g_lables_father, dim=0)
                    g_num_cat_father = len(g_lables_father_set)
                    g_lables_son = labels_source[:, self.G - 2 - i_g]
                    g_feat_btnk = out_feat_btnk[i_g + 1]
                    g_diff_sa_diff_g_dis_com_all = torch.zeros(g_num_cat_father, dtype=torch.float32, device='cuda')
                    for i_g_num_cat_father in range(g_num_cat_father):
                        g_i_cat_father = g_lables_father_set[i_g_num_cat_father]
                        g_i_feat_index_father = (g_lables_father == g_i_cat_father).nonzero(as_tuple=True)[
                            0]
                        g_i_lable_son = torch.index_select(g_lables_son, 0,
                                                           g_i_feat_index_father)
                        g_i_lables_son_set = torch.unique(g_i_lable_son, dim=0)
                        g_i_num_cat_son = len(g_i_lables_son_set)
                        if g_i_num_cat_son > 1:
                            pass
                        else:
                            continue
                        diff_sa_diff_g_com_pttp_all = torch.zeros((g_i_num_cat_son, feat_part[0]), dtype=torch.float32,
                                                                  device='cuda')
                        for i_g_son_cat in range(g_i_num_cat_son):
                            g_i_cat_son = g_i_lables_son_set[i_g_son_cat]
                            g_i_feat_index_son = (g_lables_son == g_i_cat_son).nonzero(as_tuple=True)[0]
                            g_i_son_cat_feat = torch.index_select(g_feat_btnk, 0, g_i_feat_index_son)
                            g_i_son_cat_feat_com = g_i_son_cat_feat[:, :feat_part[0], :]
                            diff_sa_diff_g_com_pttp_all[i_g_son_cat] = torch.mean(
                                torch.mean(g_i_son_cat_feat_com, dim=2), dim=0)
                            all_num_cat += g_i_num_cat_son

                        g_diff_sa_diff_g_dis_com_all[i_g_num_cat_father] = feat_sim_cos_T1(
                            diff_sa_diff_g_com_pttp_all, method='diag-I')

                    diff_sa_diff_g_dis_com_all[i_g] = torch.sum(g_diff_sa_diff_g_dis_com_all)

                loss_diff_sa_diff_g_com = torch.sum(diff_sa_diff_g_dis_com_all) / all_num_cat

                if self.args.b_diff_sa_diff_g_com is True:
                    total_loss = total_loss - loss_diff_sa_diff_g_com * self.args.c_diff_sa_diff_g_com
                else:
                    pass

            else:
                loss_diff_sa_diff_g_com = torch.tensor(0.0)
            ###################################################################################################

            self.loss['loss_same_sa_diff_g_com'] += loss_same_sa_diff_g_com.item()
            self.loss['loss_diff_sa_same_g_pvt'] += loss_diff_sa_same_g_pvt.item()
            self.loss['loss_diff_sa_diff_g_com'] += loss_diff_sa_diff_g_com.item()

        else:
            pass
        self.loss['total_loss'] += total_loss.item()

        return total_loss, self.loss