from ctpn.model import VGGTrainModel
import tensorflow as tf
from lib.fast_rcnn.config import cfg, cfg_from_file
from util.dl_components import load, save
from dataset.dataset import Dataset
from util.timer import Timer
import numpy as np
from lib.fast_rcnn.train import get_training_roidb
from lib.datasets.factory import get_imdb
from lib.roi_data_layer import roidb as rdl_roidb
from lib.fast_rcnn.train import get_data_layer

DEBUG = True

def train(max_step=50000, pretrained_model=None, restore=None, output_dir='./trained_model'):
    cfg_from_file('/home/give/PycharmProjects/MyCTPN/ctpn/text.yml')
    cfg.TRAIN.DISPLAY = 1
    imdb = get_imdb('voc_2007_trainval')
    print('Loaded dataset `{:s}` for training'.format(imdb.name))
    roidb = get_training_roidb(imdb)

    print('Computing bounding-box regression targets...')
    if cfg.TRAIN.BBOX_REG:
        bbox_means, bbox_stds = rdl_roidb.add_bbox_regression_targets(roidb)
    print('done')
    data_layer = get_data_layer(roidb, imdb.num_classes)

    vggModel = VGGTrainModel(trainable=True)
    dataset = Dataset('/home/give/Game/OCR/data/ICDAR2017/img', '/home/give/Game/OCR/data/ICDAR2017/txt')
    train_generator = dataset.train_generator
    val_generator = dataset.val_generator
    total_loss, model_loss, rpn_regression_l1_loss, rpn_cross_entropy_loss, regularization_loss = vggModel.build_loss()
    tf.summary.scalar('total_loss', total_loss)
    tf.summary.scalar('model_loss', model_loss)
    tf.summary.scalar('regularization_loss', regularization_loss)

    global_step = tf.Variable(0, trainable=False)
    lr = tf.Variable(cfg.TRAIN.LEARNING_RATE, trainable=False)
    if cfg.TRAIN.SOLVER == 'Adam':
        opt = tf.train.AdamOptimizer(cfg.TRAIN.LEARNING_RATE)
    elif cfg.TRAIN.SOLVER == 'RMS':
        opt = tf.train.RMSPropOptimizer(cfg.TRAIN.LEARNING_RATE)
    else:
        # lr = tf.Variable(0.0, trainable=False)
        momentum = cfg.TRAIN.MOMENTUM
        opt = tf.train.MomentumOptimizer(lr, momentum)
    with_clip = True
    if with_clip:
        tvars = tf.trainable_variables()
        grads, norm = tf.clip_by_global_norm(tf.gradients(total_loss, tvars), 10.0)
        train_op = opt.apply_gradients(list(zip(grads, tvars)), global_step=global_step)
    else:
        train_op = opt.minimize(total_loss, global_step=global_step)
    # tf_config = tf.ConfigProto(allow_soft_placement=True)
    # tf_config.gpu_options.allow_growth = True
    # tf_config.gpu_options.per_process_gpu_memory_fraction = 0.75
    start = 0
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(max_to_keep=5)
        if pretrained_model is not None and restore is None:
            try:
                print(('Loading pretrained model '
                       'weights from {:s}').format(pretrained_model))
                load(pretrained_model, sess, saver, ignore_missing=True)
            except:
                raise Exception('Check your pretrained model {:s}'.format(pretrained_model))
        if pretrained_model is None and restore is not None:
            try:
                print(('Loading pretrained model '
                       'weights from {:s}').format(restore))
                global_step_restore = load(restore, sess, saver)
                sess.run(global_step.assign(global_step_restore))
                start = global_step_restore
            except:
                raise Exception('Check your pretrained model {:s}'.format(restore))
        summary_op = tf.summary.merge_all()
        timer = Timer()
        for iter in range(start, max_step):
            timer.tic()
            imageGT = train_generator.next()
            blobs = data_layer.forward()
            fetch_list = [train_op, total_loss, model_loss, rpn_regression_l1_loss, rpn_cross_entropy_loss,
                          regularization_loss, summary_op]
            feed_dict_obj = {
                vggModel.input_img: imageGT.re_imgs,
                vggModel.input_gt: imageGT.gt_bboxes,
                vggModel.input_img_info: imageGT.im_info,
                vggModel.input_is_hard: imageGT.is_hard,
                vggModel.input_notcare: np.reshape(imageGT.notcare, [-1, 4]),
                vggModel.input_keepprob: 0.5
            }
            _, total_loss_val, model_loss_val, rpn_regression_l1_loss_value, rpn_cross_entropy_loss_value, regularization_loss_val, summary_op_value = sess.run(
                fetch_list, feed_dict=feed_dict_obj)
            if DEBUG:
                outputs = sess.run(
                    [vggModel.input_img, vggModel.conv1_1, vggModel.conv1_2, vggModel.conv2_1, vggModel.conv2_2,
                     vggModel.conv3_1,
                     vggModel.conv3_2, vggModel.conv3_3, vggModel.conv4_1, vggModel.conv4_2, vggModel.conv4_3,
                     vggModel.conv5_1, vggModel.conv5_2, vggModel.conv5_3, vggModel.rpn_conv, vggModel.lstm_o,
                     vggModel.lstm_bilstm, vggModel.lstm_fc,
                     vggModel.rpn_bbox_score, vggModel.rpn_bbox_pred, vggModel.rpn_data[0], vggModel.rpn_data[1], vggModel.rpn_data[2], vggModel.rpn_data[3], vggModel.rpn_cls_prob], feed_dict=feed_dict_obj)
                layer_names = [
                    'conv1_1',
                    'conv1_2',
                    'conv2_1',
                    'conv2_2',
                    'conv3_1',
                    'conv3_2',
                    'conv3_3',
                    'conv4_1',
                    'conv4_2',
                    'conv4_3',
                    'conv5_1',
                    'conv5_2',
                    'conv5_3',
                ]
                weights_biases_names = []
                for i in range(13):
                    with tf.variable_scope(layer_names[i], reuse=True):
                        weights_biases_names.append(tf.get_variable('weights'))
                        weights_biases_names.append(tf.get_variable('biases'))
                # tf.gradients()
                parameters_output = sess.run(weights_biases_names, feed_dict=feed_dict_obj)
                for i in range(13):
                    print '%s max is %.4f, min is %.4f' % (layer_names[i], np.max(outputs[i+1]), np.min(outputs[i+1]))
                    print 'responding parameters max is %.4f, min is %.4f' % (
                    np.max(parameters_output[i]), np.min(parameters_output[i]))
                print 'rpn_conv', np.max(outputs[14]), np.min(outputs[14])
                print 'lstm_o', np.max(outputs[15]), np.min(outputs[15])
                print 'lstm_bilstm', np.max(outputs[16]), np.min(outputs[16])
                print 'lstm_fc', np.max(outputs[17]), np.min(outputs[17])
                print 'rpn bbox score ', np.max(outputs[18]), np.min(outputs[18])
                print 'rpn bbox pre', np.max(outputs[19]), np.min(outputs[19])
                print 'rpn_labels', np.max(outputs[20]), np.min(outputs[20])
                print 'rpn_bbox_targets', np.max(outputs[21]), np.min(outputs[21])
                print 'rpn_bbox_inside_weights', np.max(outputs[22]), np.min(outputs[22])
                print 'rpn_bbox_outside_weights', np.max(outputs[23]), np.min(outputs[23])
                print 'rpn_cls_prob', np.max(outputs[24]), np.min(outputs[24])
                # 1717
            _diff_time = timer.toc(average=False)
            if (iter) % (cfg.TRAIN.DISPLAY) == 0:
                print(
                'iter: %d / %d, total loss: %.4f, model loss: %.4f, rpn_loss_cls: %.4f, rpn_loss_box: %.4f, lr: %f' % \
                (iter, max_step, total_loss_val, model_loss_val, rpn_cross_entropy_loss_value,
                 regularization_loss_val, lr.eval()))
                print('speed: {:.3f}s / iter'.format(_diff_time))
            if np.isnan(total_loss_val):
                label_value, score_value, rpn_cross_entropy_loss_value = sess.run(
                    [vggModel.rpn_data[0], vggModel.rpn_cls_score_reshape, vggModel.rpn_cross_entropy_loss], feed_dict={
                        vggModel.input_img: imageGT.re_imgs,
                        vggModel.input_gt: imageGT.gt_bboxes,
                        vggModel.input_img_info: imageGT.im_info,
                        vggModel.input_is_hard: imageGT.is_hard,
                        vggModel.input_notcare: np.reshape(imageGT.notcare, [-1, 4]),
                        vggModel.input_keepprob: 0.5
                    })
                print 'label value is \n', label_value
                print 'score value is \n', score_value
                print 'rpn_cross_entropy_loss_value is \n', rpn_cross_entropy_loss_value
                # print 'label value is \n', label_value
                assert not np.isnan(total_loss_val)
            if (iter + 1) % cfg.TRAIN.SNAPSHOT_ITERS == 0:
                last_snapshot_iter = iter
                save(sess, saver=saver, output_dir=output_dir, prefix='VGG', infix='_', iter_index=last_snapshot_iter)

if __name__ == '__main__':
    train(restore='/home/give/PycharmProjects/MyCTPN/ctpn/trained_model')