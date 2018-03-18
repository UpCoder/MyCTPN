# -*- coding=utf-8 -*-
import numpy as np
import cv2
from util.processing import resize_points, resolve_points, draw_polygon_image, resolve_text, draw_rects_image, read_imgs_gtfiles


class ImageGT:
    '''
    代表了Image和ground truth的组合对儿
    '''
    def __init__(self, img_files, gtline_files, limit_min=600, limit_max=800):
        '''
        初始化该对象
        :param imgs: images N * x*x * 3
        :param gtlines: N*D [['955,399,2862,412,2856,732,958,650,Latin,PRO-EXPERT\r\n']]
        :param limit_min: 图像的最小边的长度
        :param limit_max: 图像最大边的长度
        '''
        print img_files, gtline_files
        self.imgs, self.gtlines = read_imgs_gtfiles(img_files, gtline_files)
        self.limit_min = limit_min
        self.limit_max = limit_max
        self.points = resolve_points(self.gtlines[0])
        self.re_imgs, self.re_points, self.re_scales = self.resize_img()
        self.im_sizes = [np.array(re_img).shape[0:2] for re_img in self.re_imgs]
        self.gt_texts = resolve_text(self.gtlines[0])
        self.notcare = []
        cur_notcare = []
        for index, text in enumerate(self.gt_texts):
            if text == '###':
                cur_notcare.append(self.re_points[0][index])
        self.notcare.append(cur_notcare)

        self.re_points = np.squeeze(self.re_points)
        # self.gt = np.concatenate((self.re_points, np.ones([len(self.re_points), 1])), axis=1)
        self.gt_bboxes = self.re_points
        self.is_hard = [False] * len(self.gt_bboxes)
        # print np.shape(self.im_sizes), np.shape(self.re_scales)
        self.im_info = np.concatenate((self.im_sizes, self.re_scales), axis=1)
        # print np.shape(self.imgs), np.shape(self.re_points), np.shape(self.im_info)

    def resize_img(self, show=False):
        imgs = []
        points = []
        scales = []
        for index, img in enumerate(self.imgs):
            img = np.array(img)
            img_size = img.shape
            img_size_min = min(img_size[0:2])
            img_size_max = max(img_size[0:2])

            # 最小的边的长度不少于800
            im_scale = float(self.limit_min) / float(img_size_min)
            # 放缩以后，最大的边的长度不超过1200
            if np.round(im_scale * img_size_max) > self.limit_max:
                im_scale = float(self.limit_max) / float(img_size_max)
            re_im = cv2.resize(img, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)
            re_size = re_im.shape
            cur_points = resolve_points(self.gtlines[index])
            re_cur_points = resize_points(cur_points, img_size, re_size)
            if show:
                draw_polygon_image(img, cur_points)
                draw_rects_image(re_im, re_cur_points)
            imgs.append(re_im)
            points.append(re_cur_points)
            scales.append([im_scale])
        return imgs, points, scales
