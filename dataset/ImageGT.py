# -*- coding=utf-8 -*-
import numpy as np
import cv2
from util.processing import resize_points, resolve_points, draw_rects_image, resolve_text


class ImageGT:
    '''
    代表了Image和ground truth的组合对儿
    '''
    def __init__(self, imgs, gtlines, limit_min=600, limit_max=800):
        '''
        初始化该对象
        :param imgs: images N * x*x * 3
        :param gtlines: N*D [['955,399,2862,412,2856,732,958,650,Latin,PRO-EXPERT\r\n']]
        :param limit_min: 图像的最小边的长度
        :param limit_max: 图像最大边的长度
        '''
        self.imgs = imgs
        self.gtlines = gtlines
        self.limit_min = limit_min
        self.limit_max = limit_max
        self.re_imgs, self.re_points = self.resize_img()
        self.im_info = [np.array(re_img).shape[0:2] for re_img in self.re_imgs]
        print 'im_info is ', self.im_info
        self.gt_texts = resolve_text(self.gtlines[0])
        self.gt_ishard = [True if text == '###' else False for text in self.gt_texts]

    def resize_img(self, show=False):
        imgs = []
        points = []
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
            re_cur_points = resize_points(points, img_size, re_size)
            if show:
                draw_rects_image(img, cur_points)
                draw_rects_image(re_im, re_cur_points)
            imgs.append(re_im)
            points.append(re_cur_points)
        return imgs, points