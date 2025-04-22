# -*- coding:utf-8 -*-

import torch


def test_target(loader, model, criterion, args):

    with torch.no_grad():
        iter_val = [iter(loader['val' + str(i)]) for i in range(args.val_times)]

        acc_dict_test_all = []
        loss_dict_test_all = []

        for j in range(args.val_times):
            iter_time = 0
            criterion.init_loss_and_oth()
            for i in range(len(loader['val' + str(j)])):
                iter_time += 1
                data = next(iter_val[j])
                inputs = data[0].to(args.device)
                labels = data[1]
                logits_fine, logits_coarse, _, out_feat_btnk, _, _, _ = model(inputs, 1)

                total_loss_test_val, loss_dict_test_val= criterion(
                                                                    labels,
                                                                    logits_fine,
                                                                    logits_coarse,
                                                                    out_feat_btnk,
                                                                    1)

            loss_dict_test_all.append(loss_dict_test_val.copy())
            criterion.init_loss_and_oth()

            acc_dict_test_all.append(criterion.output_acc()[0])

        acc_dict_test = {**{f'{k}': 0.0 for k in acc_dict_test_all[0].keys()}}
        loss_dict_test = {**{f'{k}': 0.0 for k in loss_dict_test_val.keys()}}

        for ii in range(args.val_times):
            for jj in acc_dict_test_all[ii].keys():
                acc_dict_test[jj] += acc_dict_test_all[ii][jj].item()
            for jj in loss_dict_test_all[ii].keys():
                loss_dict_test[jj] += loss_dict_test_all[ii][jj].item() / iter_time

        acc_dict_test = {**{f'test_{k}': v / args.val_times for k, v in acc_dict_test.items()}}
        acc_dict_test['test_acc_ave'] = sum([value_i for value_i in acc_dict_test.values()]) / len(acc_dict_test)
        loss_dict_test = {**{f'test_{k}': v / args.val_times for k, v in loss_dict_test.items()}}

    return acc_dict_test, loss_dict_test

