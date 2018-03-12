from ctpn.model import VGGTrainModel
import tensorflow as tf
from lib.fast_rcnn.config import cfg
from util.dl_components import load, save
from dataset.dataset import Dataset
from util.timer import Timer
import numpy as np


def train(max_step=50000, pretrained_model=None, restore=False, output_dir='./trained_model'):
    vggModel = VGGTrainModel(trainable=True)
    dataset = Dataset('/home/give/Game/OCR/data/ICDAR2017/img_test', '/home/give/Game/OCR/data/ICDAR2017/txt_test')
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
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        if pretrained_model is not None and not restore:
            try:
                print(('Loading pretrained model '
                       'weights from {:s}').format(pretrained_model))
                load(pretrained_model, sess, True)
            except:
                raise 'Check your pretrained model {:s}'.format(pretrained_model)
        summary_op = tf.summary.merge_all()
        timer = Timer()
        saver = tf.train.Saver(max_to_keep=5)
        for iter in range(max_step):
            timer.tic()
            imageGT = train_generator.next()
            fetch_list = [train_op, total_loss, model_loss, rpn_regression_l1_loss, rpn_cross_entropy_loss,
                          regularization_loss, summary_op]

            _, total_loss_val, model_loss_val, rpn_regression_l1_loss_value, rpn_cross_entropy_loss_value, regularization_loss_val, summary_op_value = sess.run(
                fetch_list, feed_dict={
                    vggModel.input_img: imageGT.re_imgs,
                    vggModel.input_gt: imageGT.gt,
                    vggModel.input_img_info: imageGT.im_info,
                    vggModel.input_is_hard: imageGT.is_hard,
                    vggModel.input_notcare: np.reshape(imageGT.notcare, [-1, 4]),
                    vggModel.input_keepprob: 0.5
                })
            _diff_time = timer.toc(average=False)
            if (iter) % (cfg.TRAIN.DISPLAY) == 0:
                print(
                'iter: %d / %d, total loss: %.4f, model loss: %.4f, rpn_loss_cls: %.4f, rpn_loss_box: %.4f, lr: %f' % \
                (iter, max_step, total_loss_val, model_loss_val, rpn_cross_entropy_loss_value,
                 regularization_loss_val, lr.eval()))
                print('speed: {:.3f}s / iter'.format(_diff_time))

            if (iter + 1) % cfg.TRAIN.SNAPSHOT_ITERS == 0:
                last_snapshot_iter = iter
                save(sess, saver=saver, output_dir=output_dir, prefix='VGG', infix='_', iter_index=last_snapshot_iter)

if __name__ == '__main__':
    train()