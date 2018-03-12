# -*- coding=utf-8 -*-
import cv2
import numpy as np
import os
from PIL import Image, ImageDraw
import math


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
        pt_x = np.zeros((4, 1))
        pt_y = np.zeros((4, 1))
        # 重新调整坐标，因为前面有对图像的resize
        pt_x[0, 0] = int(float(point[0]) / img_size[1] * resizes[1])
        pt_y[0, 0] = int(float(point[1]) / img_size[0] * resizes[0])
        pt_x[1, 0] = int(float(point[2]) / img_size[1] * resizes[1])
        pt_y[1, 0] = int(float(point[3]) / img_size[0] * resizes[0])
        pt_x[2, 0] = int(float(point[4]) / img_size[1] * resizes[1])
        pt_y[2, 0] = int(float(point[5]) / img_size[0] * resizes[0])
        pt_x[3, 0] = int(float(point[6]) / img_size[1] * resizes[1])
        pt_y[3, 0] = int(float(point[7]) / img_size[0] * resizes[0])

        ind_x = np.argsort(pt_x, axis=0)
        pt_x = pt_x[ind_x]
        pt_y = pt_y[ind_x]

        if pt_y[0] < pt_y[1]:
            pt1 = (pt_x[0], pt_y[0])
            pt3 = (pt_x[1], pt_y[1])
        else:
            pt1 = (pt_x[1], pt_y[1])
            pt3 = (pt_x[0], pt_y[0])

        if pt_y[2] < pt_y[3]:
            pt2 = (pt_x[2], pt_y[2])
            pt4 = (pt_x[3], pt_y[3])
        else:
            pt2 = (pt_x[3], pt_y[3])
            pt4 = (pt_x[2], pt_y[2])

        xmin = int(min(pt1[0], pt2[0]))
        ymin = int(min(pt1[1], pt2[1]))
        xmax = int(max(pt2[0], pt4[0]))
        ymax = int(max(pt3[1], pt4[1]))

        if xmin < 0:
            xmin = 0
        if xmax > resizes[1] - 1:
            xmax = resizes[1] - 1
        if ymin < 0:
            ymin = 0
        if ymax > resizes[0] - 1:
            ymax = resizes[0] - 1

        width = xmax - xmin
        height = ymax - ymin

        # reimplement
        step = 16.0
        x_left = []
        x_right = []
        x_left.append(xmin)
        x_left_start = int(math.ceil(xmin / 16.0) * 16.0)
        if x_left_start == xmin:
            x_left_start = xmin + 16
        for i in np.arange(x_left_start, xmax, 16):
            x_left.append(i)
        x_left = np.array(x_left)

        x_right.append(x_left_start - 1)
        for i in range(1, len(x_left) - 1):
            x_right.append(x_left[i] + 15)
        x_right.append(xmax)
        x_right = np.array(x_right)

        idx = np.where(x_left == x_right)
        x_left = np.delete(x_left, idx, axis=0)
        x_right = np.delete(x_right, idx, axis=0)
        for i in range(len(x_left)):
            res_points.append([int(x_left[i]), int(ymin), int(x_right[i]), int(ymax)])
    return res_points


def show_image(img):
    img = Image.fromarray(img)
    img.show()


def draw_polygon_image(img, rects):
    img =Image.fromarray(img)
    draw = ImageDraw.Draw(img)

    for rect in rects:
        print rect, np.shape(img)
        draw.polygon(tuple(rect), 'olive', 'hotpink')
    img.show()


def draw_rects_image(img, rects):
    img =Image.fromarray(img)
    draw = ImageDraw.Draw(img)

    for rect in rects:
        print rect, np.shape(img)
        draw.rectangle(rect, 'olive', 'hotpink')
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