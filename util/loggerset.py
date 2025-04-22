# -*- coding:utf-8 -*-
"""*****************************************************************************
Authors: Yu wenlong  and  DRAGON_501
Description:
Functions: 
Input: 
Output: 
Note:

Link: 

*************************************Import***********************************"""
import os
import logging
from util.misc import num_coarse_cate_dataset
"""**********************************Import***********************************"""
'''***************************************************************************'''


def basicset_logger(args, i_time, all_time, loggerpredix=''):
    args.document_root = os.getcwd()

    if args.b_part_oth is False:
        args.c_part_oth = 'Fs'
    if args.b_same_sa_diff_g_com is False:
        args.c_same_sa_diff_g_com = 'Fs'
    if args.b_diff_sa_same_g_pvt is False:
        args.c_diff_sa_same_g_pvt = 'Fs'
    if args.b_diff_sa_diff_g_com is False:
        args.c_diff_sa_diff_g_com = 'Fs'

    if args.granu_num is None:
        args.granu_num = num_coarse_cate_dataset[args.dataset] + 1
    args.g = args.granu_num
    args.gc = args.granu_num - 1

    if args.resume is not None:
        args.output_root = args.resume
        args.output_dir = args.resume
        args.output_path = args.output_dir
        args.log_dir = args.output_dir

    else:
        args.resume = None
        base_docu_name = '{}_b{}lr{}g{}_fn{}fd{}_ssdgc{}__dssgp{}_dsdgc{}_otp{}_{}' \
                .format(args.train_type, args.batchsize, args.lr, args.g,
                        args.feat_num, args.feat_dim,
                        args.c_same_sa_diff_g_com,
                        args.c_diff_sa_same_g_pvt,
                        args.c_diff_sa_diff_g_com,
                        args.c_part_oth,
                        args.other)

        if args.train_type == 'tral':
            args.log_file_name_1 = '{}_{}'.format(base_docu_name, args.time_all_start)

        args.output_root = os.path.join(args.document_root, 'output')

        args.domain_name = args.source + '2' + args.target

        if args.out_forder_type == 'o':
            if len(args.out_forder_name) == 0 or args.out_forder_name is None:
                pass
            else:
                args.output_root = os.path.join(args.output_root, args.out_forder_name)

        args.output_path = os.path.join(args.output_root, args.dataset)
        args.output_path = os.path.join(args.output_path, args.backbone)

        if args.out_forder_type == 'n':
            if len(args.out_forder_name) == 0 or args.out_forder_name is None:
                args.output_dir = os.path.join(args.output_dir, 'allothers')
            else:
                args.output_dir = os.path.join(args.output_path, args.out_forder_name)
        else:
            args.output_dir = args.output_path

        args.output_path = os.path.join(args.output_dir, args.log_file_name_1)

        if args.train_type == 'tral':
            args.log_file_name = args.log_file_name_1 + '_' + args.domain_name + '_' + str(i_time)
            args.output_dir = os.path.join(args.output_path, args.log_file_name)

        os.makedirs(args.output_dir, exist_ok=True)

        args.log_dir = args.output_dir
        os.makedirs(args.log_dir, exist_ok=True)

    # times = time.strftime('%Y-%m-%d-%H-%M-%S')
    if loggerpredix != '':
        log_file_log = f'{args.log_file_name}_{loggerpredix}.log'
    else:
        log_file_log = f'{args.log_file_name}.log'
    logger = logging.getLogger(__name__)
    # logger = logging.getLogger()
    logger.handlers = []
    logger.setLevel(level=logging.DEBUG)
    handler = logging.FileHandler(os.path.join(args.log_dir, log_file_log))
    handler.setLevel(logging.DEBUG)
    han_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(han_formatter)
    logger.addHandler(handler)
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    con_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console.setFormatter(con_formatter)
    logger.addHandler(console)
    args.log_file_log = os.path.join(args.log_dir, log_file_log)
    return logger


def log_init_config(args, model, all_times):

    args.logger.info('------------------------------------------------------------------------------------------------')
    args.logger.info('**********************************************************************************************')
    n_parameters = {}
    n_parameters['all'] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    args.logger.info('all number of params: {}M\n'.format(n_parameters['all'] / 1000000))

    args_all = '*****************************************Training args:**********************************************\n'
    for k, v in sorted(vars(args).items()):
        args_all += str(k) + '=' + str(v) + '\n'
    args.logger.info(args_all)

    args.logger.info('************************************************************************************************')
    args.logger.info('model:\n')
    args.logger.info(model)
    args.logger.info('********************************************************************************************\n\n')
