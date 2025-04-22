# from __future__ import print_function, division

import torch
import numpy as np
from sklearn.preprocessing import StandardScaler
import random
from PIL import Image
import torch.utils.data as data
import os
import os.path
import warnings
import os.path as osp
from collections import OrderedDict
import shutil


class TextData():
    def __init__(self, text_file, label_file, source_batch_size=64, target_batch_size=64, val_batch_size=4):
        all_text = np.load(text_file)
        self.source_text = all_text[0:92664, :]
        self.target_text = all_text[92664:, :]
        self.val_text = all_text[0:92664, :]
        all_label = np.load(label_file)
        self.label_source = all_label[0:92664, :]
        self.label_target = all_label[92664:, :]
        self.label_val = all_label[0:92664, :]
        self.scaler = StandardScaler().fit(all_text)
        self.source_id = 0
        self.target_id = 0
        self.val_id = 0
        self.source_size = self.source_text.shape[0]
        self.target_size = self.target_text.shape[0]
        self.val_size = self.val_text.shape[0]
        self.source_batch_size = source_batch_size
        self.target_batch_size = target_batch_size
        self.val_batch_size = val_batch_size
        self.source_list = random.sample(range(self.source_size), self.source_size)
        self.target_list = random.sample(range(self.target_size), self.target_size)
        self.val_list = random.sample(range(self.val_size), self.val_size)
        self.feature_dim = self.source_text.shape[1]

    def next_batch(self, train=True):
        data = []
        label = []
        if train:
            remaining = self.source_size - self.source_id
            start = self.source_id
            if remaining <= self.source_batch_size:
                for i in self.source_list[start:]:
                    data.append(self.source_text[i, :])
                    label.append(self.label_source[i, :])
                    self.source_id += 1
                self.source_list = random.sample(range(self.source_size), self.source_size)
                self.source_id = 0
                for i in self.source_list[0:(self.source_batch_size - remaining)]:
                    data.append(self.source_text[i, :])
                    label.append(self.label_source[i, :])
                    self.source_id += 1
            else:
                for i in self.source_list[start:start + self.source_batch_size]:
                    data.append(self.source_text[i, :])
                    label.append(self.label_source[i, :])
                    self.source_id += 1
            remaining = self.target_size - self.target_id
            start = self.target_id
            if remaining <= self.target_batch_size:
                for i in self.target_list[start:]:
                    data.append(self.target_text[i, :])
                    # no target label
                    # label.append(self.label_target[i, :])
                    self.target_id += 1
                self.target_list = random.sample(range(self.target_size), self.target_size)
                self.target_id = 0
                for i in self.target_list[0:self.target_batch_size - remaining]:
                    data.append(self.target_text[i, :])
                    # label.append(self.label_target[i, :])
                    self.target_id += 1
            else:
                for i in self.target_list[start:start + self.target_batch_size]:
                    data.append(self.target_text[i, :])
                    # label.append(self.label_target[i, :])
                    self.target_id += 1
        else:
            remaining = self.val_size - self.val_id
            start = self.val_id
            if remaining <= self.val_batch_size:
                for i in self.val_list[start:]:
                    data.append(self.val_text[i, :])
                    label.append(self.label_val[i, :])
                    self.val_id += 1
                self.val_list = random.sample(range(self.val_size), self.val_size)
                self.val_id = 0
                for i in self.val_list[0:self.val_batch_size - remaining]:
                    data.append(self.val_text[i, :])
                    label.append(self.label_val[i, :])
                    self.val_id += 1
            else:
                for i in self.val_list[start:start + self.val_batch_size]:
                    data.append(self.val_text[i, :])
                    label.append(self.label_val[i, :])
                    self.val_id += 1
        data = self.scaler.transform(np.vstack(data))
        label = np.vstack(label)
        return torch.from_numpy(data).float(), torch.from_numpy(label).float()


def make_dataset(image_list, labels):
    if labels:
        len_ = len(image_list)
        images = [(image_list[i].strip(), labels[i, :]) for i in range(len_)]
    else:
        if len(image_list[0].split()) > 2:

            images = [(val.split()[0], np.array([int(la) for la in val.split()[1:]])) for val in image_list]
        else:
            images = [(val.split()[0], int(val.split()[1])) for val in image_list]
    return images


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')



def default_loader(path):
    # from torchvision import get_image_backend
    # if get_image_backend() == 'accimage':
    #    return accimage_loader(path)
    # else:
    return pil_loader(path)


class ImageList(object):
    """A generic data loader where the images are arranged in this way: ::
        root/dog/xxx.png
    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.
     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(self, image_list, labels=None, transform=None, target_transform=None,
                 loader=default_loader):
        imgs = make_dataset(image_list, labels)
        if len(imgs) == 0:
            raise (RuntimeError("Found 0 images in subfolders of: "))

        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        path, target = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def select_idx(self, img_path):

        for idx, (i_path, i_target) in enumerate(self.imgs):
            if os.path.samefile(i_path, img_path) is True:
                return idx
            else:
                if idx == len(self.imgs)-1:
                    return None

    def __len__(self):
        return len(self.imgs)


def ClassSamplingImageList(image_list, transform, return_keys=False):
    data = open(image_list).readlines()
    label_dict = {}
    for line in data:
        label_dict[int(line.split()[1])] = []
    for line in data:
        label_dict[int(line.split()[1])].append(line)
    all_image_list = {}
    for i in label_dict.keys():
        all_image_list[i] = ImageList(label_dict[i], transform=transform)
    if return_keys:
        return all_image_list, label_dict.keys()
    else:
        return all_image_list


def listdir_nohidden(path, sort=False):
    """List non-hidden items in a directory.

    Args:
         path (str): directory path.
         sort (bool): sort the items.
    """
    items = [f for f in os.listdir(path) if not f.startswith(".")]
    if sort:
        items.sort()
    return items


def check_isfile(fpath):
    """Check if the given path is a file.

    Args:
        fpath (str): file path.

    Returns:
       bool
    """
    isfile = osp.isfile(fpath)
    if not isfile:
        warnings.warn('No file found at "{}"'.format(fpath))
    return isfile


class Datum:
    """Data instance which defines the basic attributes.

    Args:
        impath (str): image path.
        label (int): class label.
        domain (int): domain label.
        classname (str): class name.
    """

    def __init__(self, impath="", label=0, domain=0, classname=""):
        assert isinstance(impath, str)
        assert check_isfile(impath)

        self._impath = impath
        self._label = label
        self._domain = domain
        self._classname = classname

    @property
    def impath(self):
        return self._impath

    @property
    def label(self):
        return self._label

    @property
    def domain(self):
        return self._domain

    @property
    def classname(self):
        return self._classname


def read_classnames(text_file):
    """Return a dictionary containing
    key-value pairs of <folder name>: <class name>.
    """
    b_ext = False
    with open(text_file, "r") as f:
        lines = f.readlines()

        line = lines[0].strip().split(" ")
        len_clasf = len(lines)
        if len(line) < 4:
            len_granu = 4
            b_ext = True
        else:
            len_granu = len(line)
            b_ext = False
    classnames_array = np.empty((len_clasf, len_granu), dtype=int)

    with open(text_file, "r") as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            line = line.strip().split(" ")
            for j, line_g in enumerate(line):
                classnames_array[i][j] = int(line_g)
            if b_ext is True:
                for jj in range(len(line), len_granu):
                    classnames_array[i][jj] = 0

    return classnames_array


def read_data(image_dir, classnames, split_dir=None):
    if split_dir is not None:
        split_dir = os.path.join(image_dir, split_dir)
    else:
        split_dir = image_dir

    image_list_new = []
    folders = sorted(f.name for f in os.scandir(split_dir) if f.is_dir())

    data_dict = {}
    class_now = {}
    classnames_now = OrderedDict()
    for label, folder in enumerate(folders):
        data_dict[int(folder)-1] = int(folder)-1
        class_now[int(folder)-1] = classnames[int(folder)-1][0]

    items = []

    for xxx, folder in enumerate(folders):
        imnames = listdir_nohidden(os.path.join(split_dir, folder))
        e_label = ' '
        for j in range(len(classnames[int(folder)-1])):
            e_label += str(classnames[int(folder)-1][j]) + ' '

        for imname in imnames:
            img_path_i = os.path.join(split_dir, folder, imname)
            img_paths = []
            if osp.isfile(img_path_i):
                img_paths.append(img_path_i)
            elif osp.isdir(img_path_i):
                img_names = listdir_nohidden(img_path_i)
                for img_name_i in img_names:
                    img_paths.append(os.path.join(img_path_i, img_name_i))
            else:
                raise 'error'

            for img_path in img_paths:
                e_all = img_path + e_label

                # item = Datum(impath=img_path, label=e_label, classname=None)
                # items.append(item)

                image_list_new.append(e_all)

    return items, classnames_now, data_dict, image_list_new


# excute this code to generate the image list for imagefolder.
# excute this code every time you change the GPU server.
if __name__ == '__main__':
    print('start')

    document_root = os.getcwd()
    file_path_all = {}

    file_path_cub = {
        "p": "./dataset_list_yuan/cub200_drawing_multi.txt",
        "c": "./dataset_list_yuan/cub200_2011_multi.txt"
    }
    file_path_birds31 = {
        "c": "./dataset_list_yuan/bird31_cub2011_multi.txt",
        "n": "./dataset_list_yuan/bird31_nabirds_list_multi.txt",
        "i": "./dataset_list_yuan/bird31_ina_list_2017_multi.txt"
    }
    file_path_comcars = {
        "s": "./dataset_list_yuan/ccars_sv_multi.txt",
        "w": "./dataset_list_yuan/ccars_web_multi.txt"
    }

    # change the file path to the dataset you want to use.
    file_path_all['cub'] = file_path_cub
    # file_path_all['birds31'] = file_path_birds31
    # file_path_all['comcars'] = file_path_comcars


    #############################################################################################
    #   extract images from bd dataset
    #############################################################################################
    # path_pro = '/data/share/bd/old/'
    # path_pro_new = '/data/share/bd1/'
    # file_path_birds31 = {
    #     "cub": "./dataset_list/bird31_cub2011_multi.txt",
    #     "nabirds": "./dataset_list/bird31_nabirds_list_multi.txt",
    #     "inalist": "./dataset_list/bird31_ina_list_2017_multi.txt"
    # }
    # domains = sorted(f.name for f in os.scandir(path_pro) if f.is_dir())
    #
    # # domains = ['nabirds']
    #
    # for xxx, domain in enumerate(domains):
    #     i = 0
    #     new_path = os.path.join(path_pro_new, domain)
    #     os.makedirs(new_path, exist_ok=True)
    #
    #     dataset_names = file_path_birds31[domain]
    #     cls_name_set = set()
    #     with open(dataset_names) as image_file:
    #         image_list = image_file.readlines()
    #         if len(image_list[0].split()) > 1:
    #             image_list_new = []
    #             image_set_names_needed = set()
    #
    #             for single_image in image_list:
    #                 e_all = single_image.split()
    #                 e_0 = e_all[0]
    #                 e_1 = e_all[1]
    #                 image_list_new.append(e_0)
    #                 cls_name_set.add(int(e_all[1]))
    #
    #                 single_names_list = e_0.split('/')
    #                 single_name = single_names_list[-1]
    #
    #                 new_cls_path = os.path.join(new_path, e_1)
    #                 os.makedirs(new_cls_path, exist_ok=True)
    #
    #                 if os.path.isfile(e_0):
    #                     shutil.copy(e_0, os.path.join(new_cls_path, single_name))
    #
    #                     image_set_names_needed.add(os.path.join(new_cls_path, single_name))
    #                     i += 1
    #
    #     #     else:
    #     #         image_list_new = image_list
    #     #
    #     #     image_set_names = set()
    #     #     cls_set_names = set()
    #     #     for single_name in image_list_new:
    #     #         single_names_list = single_name.split('/')
    #     #         image_set_names.add(single_names_list[-2] + '/' + single_names_list[-1])
    #     #         cls_set_names.add(single_names_list[-2])
    #     #
    #     #     image_list_names = list(image_set_names)
    #     # classes_names = sorted(f.name for f in os.scandir(os.path.join(path_pro, domain)) if f.is_dir())
    #     #
    #     # image_set_names_needed = set()
    #     # for class_name in classes_names:
    #     #     imgs_path = os.path.join(path_pro, domain, class_name)
    #     #     if osp.isdir(imgs_path):
    #     #         img_names = listdir_nohidden(imgs_path)
    #     #         for img_name_i in img_names:
    #     #             img_path = os.path.join(imgs_path, img_name_i)
    #     #
    #     #             new_img_cls_name = class_name + '/' + img_name_i
    #     #             if new_img_cls_name in image_list_names:
    #     #                 new_path_clses = os.path.join(new_path, class_name)
    #     #                 os.makedirs(new_path_clses, exist_ok=True)
    #     #                 image_set_names_needed.add(img_path)
    #     #                 shutil.copy(img_path, os.path.join(new_path_clses, img_name_i))
    #     #                 i += 1
    #     print(domain, len(image_set_names_needed))
    #############################################################################################
    #############################################################################################


    # #############################################################################################
    # # modify comcars web to without year information
    # #############################################################################################
    # path_pro = './data/CompCars281R/Web/data/image/'
    # path_pro_new = './data/CompCars281R/Web/data/image_new/'
    # folders = sorted(f.name for f in os.scandir(path_pro) if f.is_dir())
    # i = 0
    # for xxx, folder in enumerate(folders):
    #     imnames = listdir_nohidden(os.path.join(path_pro, folder))
    #     new_path = os.path.join(path_pro_new, folder)
    #     os.makedirs(new_path, exist_ok=True)
    #     for imname in imnames:
    #         img_path_i = os.path.join(path_pro, folder, imname)
    #         if osp.isdir(img_path_i):
    #             img_names = listdir_nohidden(img_path_i)
    #             for img_name_i in img_names:
    #                 img_path = os.path.join(img_path_i, img_name_i)
    #                 new_img_names = imname + img_name_i
    #                 shutil.copy(img_path, os.path.join(new_path, new_img_names))
    #                 i += 1
    # #############################################################################################
    # #############################################################################################

    # #############################################################################################
    # # processing comcars dataset
    # #############################################################################################
    # ori_path = './data/CompCars281R/datalist/CompCars_map.txt'
    # class_map_file = './data/CompCars281R/datalist/'
    # s_path = './data/CompCars281R/SV/sv_data/image/'
    # w_path = './data/CompCars281R/Web/data/image_new/'
    #
    # classnames = read_classnames(ori_path)
    # img_s, classnames_s, cls2idx_s, img_list_s = read_data(s_path, classnames)
    # img_w, classnames_w, cls2idx_w, img_list_w = read_data(w_path, classnames)
    #
    # with open(file_path_comcars['s'], 'w', encoding='utf-8') as fw_s:
    #     for line in range(len(img_list_s)):
    #         fw_s.write(str(img_list_s[line]) + '\n')
    # fw_s.close()
    #
    # with open(file_path_comcars['w'], 'w', encoding='utf-8') as fw:
    #     for line in range(len(img_list_w)):
    #         fw.write(str(img_list_w[line]) + '\n')
    # fw.close()
    # #############################################################################################
    # #############################################################################################

    # #############################################################################################
    # # processing iNaturalist dataset
    # #############################################################################################
    #
    # i_origin_path = '/data/share/iNaturalist/train_val2017/train_val2017/Aves/'
    # files_all = os.listdir(i_origin_path)
    # for i in range(len(files_all)):
    #     n1 = i_origin_path + files_all[i]
    #     n2 = n1.replace(' ', '_')
    #     os.rename(n1, n2)
    # #############################################################################################
    # #############################################################################################

    for key, file_path in file_path_all.items():

        for i_domain_name, i_domain_path in file_path.items():

            dataset_source = file_path[i_domain_name]
            file_name = dataset_source.split('/')[-1]

            with open(dataset_source) as image_file:
                image_list = image_file.readlines()

                if len(image_list[0].split()) > 2:
                    image_list_new = []

                    for single_image in image_list:
                        e_all = single_image.split()
                        e_0 = single_image.split()[0]
                        e_1 = single_image.split()[1:]

                        path_elements = e_0.split('/')

                        start_index = path_elements.index('data') + 1

                        if i_domain_name == 'c':
                            start_index = 5
                        elif i_domain_name == 'p':
                            start_index = 3
                        elif i_domain_name == 'n':
                            start_index = 2
                        elif i_domain_name == 'i':
                            start_index = 3
                        elif i_domain_name == 'w':
                            start_index = 3
                        elif i_domain_name == 's':
                            start_index = 3

                        e_path = e_0.split('/')

                        e_2 = e_path[start_index:]  # cub_drawing 3，cub_200_2011 5
                        root_path = '/data'
                        for i in range(len(e_2)):
                            e_3 = '/' + e_2[i]
                            root_path += e_3
                        image_new = document_root + root_path
                        # e_0 = image_new
                        for j in range(len(e_1)):
                            e_4 = ' ' + e_1[j]
                            image_new += e_4

                        e_x = image_new.split()
                        # print('over')
                        image_list_new.append(image_new)
                    os.makedirs(f'./dataset_list/', exist_ok=True)
                    with open(f'./dataset_list/{file_name}', 'w', encoding='utf-8') as fw:
                        for line in range(len(image_list_new)):
                            fw.write(str(image_list_new[line]) + '\n')
                    fw.close()
        print(f'{key} over')
    print('all over')


