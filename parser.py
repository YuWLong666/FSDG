# -*- coding:utf-8 -*-
"""*****************************************************************************
Time:    2022- 09- 22
Authors: Yu wenlong  and  DRAGON_501
Description:
Functions: 
Input: 
Output: 
Note:

Link: 

*************************************Import***********************************"""
import argparse

"""**********************************Import***********************************"""


def get_args_parser():
    parser = argparse.ArgumentParser(description='FSDG', add_help=False)
    str2bool = lambda x: x.lower() == "true"

    parser.add_argument('--goal', default='xai')
    # parser.add_argument('--other', '-ot', default='xxx')
    parser.add_argument('--other', '-ot', default='debug')
    parser.add_argument('--out_forder_name', '-ofn', default='xxx', help='')
    parser.add_argument('--out_forder_type', '-oft', default='o', help='o: all in one folder；'
                                                                       'n：folder under every dataset and bkb')

    parser.add_argument('--granu_num', '-g', default=4, type=int)

    parser.add_argument('--bn_cfdg', type=str2bool, default=True)
    parser.add_argument('--b_bkb_c', type=str2bool, default=True)
    parser.add_argument('--b_lamda', type=str2bool, default=True)

    parser.add_argument('--bn_align', action='store_false')
    parser.add_argument('--b_f_ce', type=str2bool, default=False)

    parser.add_argument('--lr', type=float, nargs='?', default=0.03, help="target dataset")
    parser.add_argument('--feat_dim', type=int, default=2048, help="")
    parser.add_argument('--feat_num', type=int, default=128, help="")
    parser.add_argument('--feat_ratio', type=str, default='50_30_20')

    parser.add_argument('--sim_method', type=str, default='no')

    parser.add_argument('--b_part_oth', '-b_po', action='store_true')
    parser.add_argument('--b_same_sa_diff_g_com', '-b_ssdgc', action='store_true')  # 2
    parser.add_argument('--b_diff_sa_same_g_pvt', '-b_dssgp', action='store_true')  # 5
    parser.add_argument('--b_diff_sa_diff_g_com', '-b_dsdgc', action='store_true')  # 6

    #
    parser.add_argument('--cnfd_type', default='no')
    parser.add_argument('--b_cndg', action='store_true')

    parser.add_argument('--oth_num', type=int, default=4)

    parser.add_argument('--c_part_oth', '-c_po', type=float, default=1)
    parser.add_argument('--c_same_sa_diff_g_com', '-c_ssdgc', type=float, default=0.05)
    parser.add_argument('--c_diff_sa_same_g_pvt', '-c_dssgp', type=float, default=1)
    parser.add_argument('--c_diff_sa_diff_g_com', '-c_dsdgc', type=float, default=0.1)

    parser.add_argument('--c_loss_entropy_source', type=float, default=0, help="target dataset")
    parser.add_argument('--c_loss_entropy_target', type=float, default=0.01, help="target dataset")

    parser.add_argument('--config', default='configfile/config.yaml')
    parser.add_argument('--train_state', default='tral')
    parser.add_argument('--train_type', '-tt', default='tral')
    parser.add_argument('--mu', type=float, default=0.9, help="")

    # parser.add_argument('--norm_type', default='T1')

    parser.add_argument('--dataset', type=str, default='cp2', choices=('cp2', 'bd', 'cars', 'in'))
    parser.add_argument('--source', type=str, default='c', help="source dataset")
    parser.add_argument('--target', type=str, default='p', help="target dataset")
    parser.add_argument('--batchsize', '-bs', type=int, default=32)
    parser.add_argument('--bs_test', type=int, default=None)

    parser.add_argument('--backbone', '-bkb', default='rn50', type=str, choices=(
        'generator',
        'rn50', 'rn101', 'rn152',
        'vit_tiny_patch16_384', 'vit_tiny_patch16_224',
        'vit_small_patch16_384', 'vit_small_patch16_224', 'vit_small_patch32_384',
        'vit_base_patch16_384', 'vit_base_patch16_224', 'vit_base_patch32_224',
        'vit_large_patch16_384', 'vit_large_patch16_224', 'vit_large_patch32_224',
        'dino_vits16', 'dino_vitb16', 'dino_resnet50',
        'mixer_b16_224_in21k',
        'asmlp-tiny', 'asmlp-small', 'asmlp-base',
        'swin_tiny_patch4_window7_224.ms_in1k'))

    parser.add_argument('--initial_smooth', type=float, nargs='?', default=0.9, help="target dataset")
    parser.add_argument('--final_smooth', type=float, nargs='?', default=0.1, help="target dataset")

    parser.add_argument('--max_iteration', '-iter', type=float, default=20000, help="20000")
    parser.add_argument('--test_interval', type=float, default=1000, help="500")
    parser.add_argument('--time_all_start', type=str)

    parser.add_argument('--smooth_stratege', type=str, nargs='?', default='e', help="smooth stratege")

    parser.add_argument("--h", default=0, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')
    parser.add_argument('--resume', help='resume from checkpoint',
        default=None
                        )

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', '-b_dis', type=str2bool, default=False)

    parser.add_argument("--times", "-t", default=3, type=int, help="Repeat times")
    parser.add_argument("--startdomain", default=0, type=int)

    parser.add_argument("--val_times", default=10, type=int, help="val Repeat times")
    parser.add_argument("--output_dir", default='', type=str, help="Repeat times")
    parser.add_argument("--b_use_tensorboard", type=bool, default=False)
    parser.add_argument("--bn_store_ckpt", type=str2bool, default=False)

    parser.add_argument('--viz', default=False)
    parser.add_argument('--eval', default=False)

    return parser


if __name__ == '__main__':
    parser = argparse.ArgumentParser('FSDG', parents=[get_args_parser()])

    args = parser.parse_args()

    print(args)




