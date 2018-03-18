from model import VGGTestModel
import tensorflow as tf
from dataset.dataset import Dataset
from lib.fast_rcnn.config import cfg
import numpy as np
from util.processing import draw_rects_image, show_image


def test_oneimage(sess, vggObj, dataObj):
    bboxs_values = sess.run([vggObj.rpn_rois], feed_dict={
        vggObj.input_img: dataObj.re_imgs,
        vggObj.input_img_info: dataObj.im_info
    })
    bboxs_values = np.squeeze(np.asarray(bboxs_values, np.float32))
    print bboxs_values, np.shape(bboxs_values)
    return bboxs_values


if __name__ == '__main__':
    model_path = '/home/give/PycharmProjects/MyCTPN/ctpn/trained_model'
    ckpt = tf.train.get_checkpoint_state(model_path)
    vggModel = VGGTestModel()
    with tf.Session() as sess:
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, ckpt.model_checkpoint_path)
        dataset = Dataset('/home/give/Game/OCR/data/ICDAR2017/img_test', '/home/give/Game/OCR/data/ICDAR2017/txt_test')
        val_generator = dataset.val_generator
        img_obj = val_generator.next()
        bbox_values = test_oneimage(sess, vggModel, img_obj)
        scores = bbox_values[:, 0]
        bbox_points = bbox_values[:, 1:5]
        bbox_points = bbox_points[scores > 0.7, :]
        print np.shape(scores), np.shape(bbox_points)
        img = np.squeeze(img_obj.re_imgs)
        img += cfg.PIXEL_MEANS
        print np.shape(img)
        show_image(np.asarray(img, np.uint8))
        draw_rects_image(np.asarray(img, np.uint8), bbox_points)