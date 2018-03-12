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
    return img


def read_imgs_gtfiles(img_files, gt_files):
    '''
    读取图像文件和ground truth
    :param img_files:
    :param gt_files:
    :return:
    '''
    imgs = []
    gts = []
    for index, img_file in enumerate(img_files):
        img = read_image(img_file)
        if img is None:
            continue
        imgs.append(img)
        gts.append(read_gtfile(gt_files[index]))
    return imgs, gts


def read_gtfile(gt_txt_path):
    '''
    将gt文件中的所有行读取出来
    :param gt_txt_path:
    :return:
    '''
    with open(gt_txt_path, 'r') as f:
        lines = f.readlines()
    return lines


def resolve_text(lines):
    texts = []
    for line in lines:
        splitted_line = line.strip().lower().split(',')
        texts.append(splitted_line[-1])
    return texts


def resolve_points(lines):
    points = []
    for line in lines:
        splitted_line = line.strip().lower().split(',')
        for i in range(8):
            splitted_line[i] = int(splitted_line[i])
        points.append((splitted_line[0:8]))
        # print len(points[0])
    return np.array(points)


def resize_points(points, img_size, resizes):
    res_points = []
    for point in points:
        cur_point = np.zeros([8, 1])
        cur_point[0, 0] = int(float(point[0]) / img_size[1] * resizes[1])
        cur_point[1, 0] = int(float(point[1]) / img_size[0] * resizes[0])
        cur_point[2, 0] = int(float(point[2]) / img_size[1] * resizes[1])
        cur_point[3, 0] = int(float(point[3]) / img_size[0] * resizes[0])
        cur_point[4, 0] = int(float(point[4]) / img_size[1] * resizes[1])
        cur_point[5, 0] = int(float(point[5]) / img_size[0] * resizes[0])
        cur_point[6, 0] = int(float(point[6]) / img_size[1] * resizes[1])
        cur_point[7, 0] = int(float(point[7]) / img_size[0] * resizes[0])
        res_points.append(cur_point)
    return res_points


def show_image(img):
    img = Image.fromarray(img)
    img.show()


def draw_rects_image(img, rects):
    img =Image.fromarray(img)
    draw = ImageDraw.Draw(img)

    for rect in rects:
        print rect, np.shape(img)
        draw.polygon(tuple(rect), 'olive', 'hotpink')
    img.show()

def resize_image_gt(image_path, gt_path):
    img = read_image(image_path)
    lines = read_gtfile(gt_path)
    print lines
    points = resolve_points(lines)
    print points
    img_size = img.shape
    img_size_min = min(img_size[0:2])
    img_size_max = max(img_size[0:2])

    # 最小的边的长度不少于800
    im_scale = float(800) / float(img_size_min)
    # 放缩以后，最大的边的长度不超过1200
    if np.round(im_scale * img_size_max) > 1200:
        im_scale = float(1200) / float(img_size_max)
    re_im = cv2.resize(img, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)
    re_size = re_im.shape
    print re_size
    re_points = resize_points(points, img_size, re_size)
    draw_rects_image(img, [points[8]])
    draw_rects_image(re_im, [re_points[8]])

if __name__ == '__main__':
    resize_image_gt('/home/give/PycharmProjects/MyCTPN/test/img_1.png', '/home/give/PycharmProjects/MyCTPN/test/gt_img_1.txt')