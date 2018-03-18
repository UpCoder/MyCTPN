# -*- coding=utf-8 -*-
import tensorflow as tf
from util.dl_components import do_conv, _modified_smooth_l1, do_maxpooling, do_bilstm_fc, lstm_fc, do_reshape
from lib.fast_rcnn.config import cfg
from lib.rpn_msr.anchor_target_layer_tf import anchor_target_layer
from lib.rpn_msr.proposal_layer_tf import proposal_layer as proposal_layer_py


def get_network(is_training):
    if is_training:
        print is_training
    else:
        print is_training

anchor_scales = cfg.ANCHOR_SCALES
_feat_stride = [16, ]

class VGGTestModel:
    def __init__(self):
        self.input_img = tf.placeholder(tf.float32, [None, None, None, 3], name='input_img')
        # [None, 3] represent the size and the scale
        self.input_img_info = tf.placeholder(tf.float32, [None, 3], name='input_img_info')
        self.model()

    def model(self):
        self.conv1_1 = do_conv(self.input_img, 'conv1_1', [3, 3], 64, [1, 1], activation_method=tf.nn.relu,
                               is_training=False, bn=True, trainable=True)
        self.conv1_2 = do_conv(self.conv1_1, 'conv1_2', [3, 3], 64, [1, 1], activation_method=tf.nn.relu,
                               is_training=False, bn=True, trainable=True)
        self.pool1 = do_maxpooling(self.conv1_2, padding='VALID', layer_name='pool1', kernel_size=[2, 2],
                                   stride_size=[2, 2])
        self.conv2_1 = do_conv(self.pool1, 'conv2_1', [3, 3], 128, [1, 1], activation_method=tf.nn.relu, trainable=True, is_training=False, bn=True)
        self.conv2_2 = do_conv(self.conv2_1, 'conv2_2', [3, 3], 128, [1, 1], activation_method=tf.nn.relu,
                               is_training=False, bn=True,trainable=True)
        self.pool2 = do_maxpooling(self.conv2_2, layer_name='pool2', padding='VALID', kernel_size=[2, 2],
                                   stride_size=[2, 2])
        self.conv3_1 = do_conv(self.pool2, 'conv3_1', kernel_size=[3, 3], filter_size=256, stride_size=[1, 1],
                               is_training=False, bn=True, activation_method=tf.nn.relu)
        self.conv3_2 = do_conv(self.conv3_1, 'conv3_2', kernel_size=[3, 3], filter_size=256, stride_size=[1, 1],
                               is_training=False, bn=True, activation_method=tf.nn.relu)
        self.conv3_3 = do_conv(self.conv3_2, 'conv3_3', kernel_size=[3, 3], filter_size=256, stride_size=[1, 1],
                               is_training=False, bn=True, activation_method=tf.nn.relu)
        self.pool3 = do_maxpooling(self.conv3_3, layer_name='pool3', padding='VALID', kernel_size=[2, 2],
                                   stride_size=[2, 2])
        self.conv4_1 = do_conv(self.pool3, layer_name='conv4_1', kernel_size=[3, 3], filter_size=512,
                               is_training=False, bn=True, stride_size=[1, 1],
                               activation_method=tf.nn.relu)
        self.conv4_2 = do_conv(self.conv4_1, layer_name='conv4_2', kernel_size=[3, 3], filter_size=512,
                               is_training=False, bn=True, stride_size=[1, 1],
                               activation_method=tf.nn.relu)
        self.conv4_3 = do_conv(self.conv4_2, layer_name='conv4_3', kernel_size=[3, 3], filter_size=512,
                               is_training=False, bn=True, stride_size=[1, 1],
                               activation_method=tf.nn.relu)
        self.pool4 = do_maxpooling(self.conv4_3, layer_name='pool4', padding='VALID', kernel_size=[2, 2],
                                   stride_size=[2, 2])
        self.conv5_1 = do_conv(self.pool4, layer_name='conv5_1', kernel_size=[3, 3], filter_size=512,
                               is_training=False, bn=True, stride_size=[1, 1],
                               activation_method=tf.nn.relu)
        self.conv5_2 = do_conv(self.conv5_1, layer_name='conv5_2', kernel_size=[3, 3], filter_size=512,
                               is_training=False, bn=True, stride_size=[1, 1],
                               activation_method=tf.nn.relu)
        self.conv5_3 = do_conv(self.conv5_2, layer_name='conv5_3', kernel_size=[3, 3], filter_size=512,
                               is_training=False, bn=True, stride_size=[1, 1],
                               activation_method=tf.nn.relu)

        # Region Proposal Network
        self.rpn_conv = do_conv(self.conv5_3, layer_name='rpn_conv', kernel_size=[3, 3], filter_size=512,
                                stride_size=[1, 1],
                                activation_method=tf.nn.relu)

        self.lstm_o, self.lstm_bilstm, self.lstm_fc = do_bilstm_fc(self.rpn_conv, layer_name='bilstm_fc', d_i=512,
                                                                   d_h=128, d_o=512, trainable=True)

        self.rpn_bbox_pred = lstm_fc(self.lstm_o, 'rpn_bbox_pred', 512, len(cfg.ANCHOR_SCALES) * 10 * 4, True)
        self.rpn_bbox_score = lstm_fc(self.lstm_o, 'rpn_bbox_score', 512, len(cfg.ANCHOR_SCALES) * 10 * 2, True)

        self.rpn_cls_score_reshape = do_reshape(self.rpn_bbox_score, 2, name='rpn_cls_score_reshape')
        self.rpn_cls_prob = tf.nn.softmax(self.rpn_cls_score_reshape)

        self.rpn_cls_prob_reshape = do_reshape(self.rpn_cls_prob, len(cfg.ANCHOR_SCALES) * 10 * 2,
                                               name='rpn_cls_prob_reshape')

        blob, bbox_delta = tf.py_func(proposal_layer_py,
                                      [self.rpn_cls_prob_reshape, self.rpn_bbox_pred, self.input_img_info, 'TEST',
                                       _feat_stride, anchor_scales], \
                                      [tf.float32, tf.float32])
        self.rpn_rois = tf.convert_to_tensor(tf.reshape(blob, [-1, 5]), name='rpn_rois')  # shape is (1 x H x W x A, 5)
        self.rpn_targets = tf.convert_to_tensor(bbox_delta, name='rpn_targets')  # shape is (1 x H x W x A, 4)


class VGGTrainModel:
    def __init__(self, trainable):
        self.input_img = tf.placeholder(tf.float32, [None, None, None, 3], name='input_img')
        # [None, 3] represent the size and the scale
        self.input_img_info = tf.placeholder(tf.float32, [None, 3], name='input_img_info')
        self.input_gt = tf.placeholder(tf.float32, [None, 4], name='input_gt')
        self.input_is_hard = tf.placeholder(tf.int32, [None], name='input_is_hard')
        self.input_notcare = tf.placeholder(tf.float32, [None, 4], name='input_notcare')
        self.input_keepprob = tf.placeholder(tf.float32, name='input_keep_probility')
        self.trainable = trainable
        self.model()

    def model(self):
        # extract feature map
        self.conv1_1 = do_conv(self.input_img, 'conv1_1', [3, 3], 64, [1, 1], activation_method=tf.nn.relu,
                               is_training=True, bn=True, trainable=True)
        self.conv1_2 = do_conv(self.conv1_1, 'conv1_2', [3, 3], 64, [1, 1], activation_method=tf.nn.relu,
                               is_training=True, bn=True, trainable=True)
        self.pool1 = do_maxpooling(self.conv1_2, padding='VALID', layer_name='pool1', kernel_size=[2, 2],
                                   stride_size=[2, 2])
        self.conv2_1 = do_conv(self.pool1, 'conv2_1', [3, 3], 128, [1, 1], activation_method=tf.nn.relu, trainable=True,
                               is_training=True, bn=True)
        self.conv2_2 = do_conv(self.conv2_1, 'conv2_2', [3, 3], 128, [1, 1], activation_method=tf.nn.relu,
                               is_training=True, bn=True,trainable=True)
        self.pool2 = do_maxpooling(self.conv2_2, layer_name='pool2', padding='VALID', kernel_size=[2, 2],
                                   stride_size=[2, 2])
        self.conv3_1 = do_conv(self.pool2, 'conv3_1', kernel_size=[3, 3], filter_size=256, stride_size=[1, 1],
                               is_training=True, bn=True,activation_method=tf.nn.relu)
        self.conv3_2 = do_conv(self.conv3_1, 'conv3_2', kernel_size=[3, 3], filter_size=256, stride_size=[1, 1],
                               is_training=True, bn=True,activation_method=tf.nn.relu)
        self.conv3_3 = do_conv(self.conv3_2, 'conv3_3', kernel_size=[3, 3], filter_size=256, stride_size=[1, 1],
                               is_training=True, bn=True, activation_method=tf.nn.relu)
        self.pool3 = do_maxpooling(self.conv3_3, layer_name='pool3', padding='VALID', kernel_size=[2, 2],
                                   stride_size=[2, 2])
        self.conv4_1 = do_conv(self.pool3, layer_name='conv4_1', kernel_size=[3, 3], filter_size=512,
                               is_training=True, bn=True,stride_size=[1, 1],
                               activation_method=tf.nn.relu)
        self.conv4_2 = do_conv(self.conv4_1, layer_name='conv4_2', kernel_size=[3, 3], filter_size=512,
                               is_training=True, bn=True,stride_size=[1, 1],
                               activation_method=tf.nn.relu)
        self.conv4_3 = do_conv(self.conv4_2, layer_name='conv4_3', kernel_size=[3, 3], filter_size=512,
                               is_training=True, bn=True,stride_size=[1, 1],
                               activation_method=tf.nn.relu)
        self.pool4 = do_maxpooling(self.conv4_3, layer_name='pool4', padding='VALID', kernel_size=[2, 2],
                                   stride_size=[2, 2])
        self.conv5_1 = do_conv(self.pool4, layer_name='conv5_1', kernel_size=[3, 3], filter_size=512,
                               is_training=True, bn=True,stride_size=[1, 1],
                               activation_method=tf.nn.relu)
        self.conv5_2 = do_conv(self.conv5_1, layer_name='conv5_2', kernel_size=[3, 3], filter_size=512,
                               is_training=True, bn=True,stride_size=[1, 1],
                               activation_method=tf.nn.relu)
        self.conv5_3 = do_conv(self.conv5_2, layer_name='conv5_3', kernel_size=[3, 3], filter_size=512,
                               is_training=True, bn=True,stride_size=[1, 1],
                               activation_method=tf.nn.relu)

        # Region Proposal Network
        self.rpn_conv = do_conv(self.conv5_3, layer_name='rpn_conv', kernel_size=[3, 3], filter_size=512, stride_size=[1, 1],
                           activation_method=tf.nn.relu)

        self.lstm_o, self.lstm_bilstm, self.lstm_fc = do_bilstm_fc(self.rpn_conv, layer_name='bilstm_fc', d_i=512,
                                                                   d_h=128, d_o=512, trainable=True)

        self.rpn_bbox_pred = lstm_fc(self.lstm_o, 'rpn_bbox_pred', 512, len(cfg.ANCHOR_SCALES) * 10 * 4, True)
        self.rpn_bbox_score = lstm_fc(self.lstm_o, 'rpn_bbox_score', 512, len(cfg.ANCHOR_SCALES) * 10 * 2, True)

        rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights = tf.py_func(anchor_target_layer,
                              [self.rpn_bbox_score, self.input_gt, self.input_is_hard, self.input_notcare,
                               self.input_img_info, _feat_stride, anchor_scales],
                              [tf.float32, tf.float32, tf.float32, tf.float32])
        rpn_labels = tf.convert_to_tensor(tf.cast(rpn_labels, tf.int32),
                                          name='rpn_labels')  # shape is (1 x H x W x A, 2)
        rpn_bbox_targets = tf.convert_to_tensor(rpn_bbox_targets,
                                                name='rpn_bbox_targets')  # shape is (1 x H x W x A, 4)
        rpn_bbox_inside_weights = tf.convert_to_tensor(rpn_bbox_inside_weights,
                                                       name='rpn_bbox_inside_weights')  # shape is (1 x H x W x A, 4)
        rpn_bbox_outside_weights = tf.convert_to_tensor(rpn_bbox_outside_weights,
                                                        name='rpn_bbox_outside_weights')  # shape is (1 x H x W x A, 4)
        self.rpn_data = []
        self.rpn_data.append(rpn_labels)
        self.rpn_data.append(rpn_bbox_targets)
        self.rpn_data.append(rpn_bbox_inside_weights)
        self.rpn_data.append(rpn_bbox_outside_weights)

        self.rpn_cls_score_reshape = do_reshape(self.rpn_bbox_score, 2, name='rpn_cls_score_reshape')
        self.rpn_cls_prob = tf.nn.softmax(self.rpn_cls_score_reshape)

    def smooth_l1_dist(self, deltas, sigma2=9.0, name='smooth_l1_dist'):
        with tf.name_scope(name=name) as scope:
            deltas_abs = tf.abs(deltas)
            smoothL1_sign = tf.cast(tf.less(deltas_abs, 1.0/sigma2), tf.float32)
            return tf.square(deltas) * 0.5 * sigma2 * smoothL1_sign + \
                        (deltas_abs - 0.5 / sigma2) * tf.abs(smoothL1_sign - 1)

    def build_loss(self):
        # region proposal score loss
        rpn_cls_score = tf.reshape(self.rpn_cls_score_reshape, [-1, 2])
        rpn_label = tf.reshape(self.rpn_data[0], [-1])
        fg_keep = tf.equal(rpn_label, 1)
        # 挑选出我们在意的anchor，如果label为-1，在我们不用在意
        care_inds = tf.where(tf.not_equal(rpn_label, -1))
        rpn_cls_score = tf.gather(rpn_cls_score, care_inds)
        rpn_label = tf.gather(rpn_label, care_inds)
        self.rpn_cross_entropy_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=rpn_label,
                                                                                     logits=rpn_cls_score)

        # bbox loss
        rpn_bbox_pred = self.rpn_bbox_pred
        rpn_bbox_target = self.rpn_data[1]
        rpn_bbox_pred = tf.gather(rpn_bbox_pred, care_inds)
        rpn_bbox_target = tf.gather(rpn_bbox_target, care_inds)
        rpn_bbox_inside_weights = tf.gather(self.rpn_data[2], care_inds)
        rpn_bbox_outside_weights = tf.gather(self.rpn_data[3], care_inds)
        rpn_loss_box_n = tf.reduce_sum(rpn_bbox_outside_weights * self.smooth_l1_dist(
            rpn_bbox_inside_weights * (rpn_bbox_pred - rpn_bbox_target)), reduction_indices=[1])

        rpn_loss_box = tf.reduce_sum(rpn_loss_box_n) / (tf.reduce_sum(tf.cast(fg_keep, tf.float32)) + 1)

        # rpn_regression_l1_loss = _modified_smooth_l1(3.0, rpn_bbox_pred, rpn_bbox_target, rpn_bbox_inside_weights, rpn_bbox_outside_weights)

        model_loss = rpn_loss_box + tf.reduce_mean(self.rpn_cross_entropy_loss)
        regularization_losses = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        total_loss = model_loss + regularization_losses
        return total_loss, model_loss, rpn_loss_box, tf.reduce_mean(
            self.rpn_cross_entropy_loss), regularization_losses

