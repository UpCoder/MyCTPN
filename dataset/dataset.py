# -*- coding=utf-8 -*-
import numpy as np
from util.processing import read_imgs_gtfiles
from util.utils import split2_train_val
import os
from Generator import GenerateBatch


class Dataset:
    def check_sort(self):
        '''
        检查排序是否正确
        :return: 正确：True; 错误：False
        '''
        for index, img_file in enumerate(self.img_files):
            _, basename = os.path.split(img_file)
            img_id = os.path.splitext(basename)[0]
            gt_id = os.path.splitext(os.path.basename(self.gt_files[index]))[0]
            if not gt_id.endswith(img_id):
                return False
        return True

    def __init__(self, img_dir, gt_dir):
        self.img_dir = img_dir
        self.gt_dir = gt_dir
        self.img_files = np.array([os.path.join(self.img_dir, file_name) for file_name in os.listdir(self.img_dir)])
        self.gt_files = np.array([os.path.join(self.gt_dir, file_name) for file_name in os.listdir(self.gt_dir)])
        inds = np.argsort(self.img_files, axis=0)
        self.img_files = self.img_files[inds]
        inds = np.argsort(self.gt_files)
        self.gt_files = self.gt_files[inds]
        if not self.check_sort():
            return
        self.img_train_files, self.gt_train_files, self.img_val_files, self.gt_val_files = split2_train_val(
            self.img_files, self.gt_files)
        self.train_imgs, self.train_gt = read_imgs_gtfiles(self.img_train_files, self.gt_train_files)
        self.val_imgs, self.val_gt = read_imgs_gtfiles(self.img_val_files, self.gt_val_files)
        self.train_generator = GenerateBatch(self.train_imgs, self.train_gt, batch_size=1).generate_next_batch()
        self.val_generator = GenerateBatch(self.val_imgs, self.val_gt, batch_size=1).generate_next_batch()


if __name__ == '__main__':
    dataset = Dataset('/home/give/Game/OCR/data/ICDAR2017/img_test', '/home/give/Game/OCR/data/ICDAR2017/txt_test')
    # for i in range(120):
    imageGT = dataset.train_generator.next()
    imageGT.resize_img(show=True)