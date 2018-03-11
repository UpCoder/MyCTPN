# -*- coding=utf-8 -*-
import cv2
import numpy as np
import os
from PIL import Image, ImageDraw


def read_image(image_path):
    '''
    读取图像文件
    :param image_path:
    :return:
    '''
    _, basename = os.path.split(image_path)
    if basename.lower().split('.')[-1] not in ['jpg', 'png']:
        return None
    img = cv2.imread(image_path)
    print np.shape(img)
    return img


def read_gtfile(gt_txt_path):
    '''
    将gt文件中的所有行读取出来
    :param gt_txt_path:
    :return:
    '''
    with open(gt_txt_path, 'r') as f:
        lines = f.readlines()
    return lines


def resize_image_gt(image_path, gt_path):
    img = read_image(image_path)
    lines = read_gtfile(gt_path)

    img_size = img.shape

if __name__ == '__main__':
    read_image('/home/give/PycharmProjects/MyCTPN/test/img_1.jpg')