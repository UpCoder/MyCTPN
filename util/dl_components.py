import tensorflow as tf
import numpy as np
import os
from lib.fast_rcnn.config import cfg


def get_weights(name, shape, initializer, trainable, regularizer=None):
    return tf.get_variable(name, dtype=tf.float32, shape=shape, initializer=initializer, trainable=trainable,
                           regularizer=regularizer)


def _modified_smooth_l1(sigma, bbox_pred, bbox_targets, bbox_inside_weights, bbox_outside_weights):
    """
        ResultLoss = outside_weights * SmoothL1(inside_weights * (bbox_pred - bbox_targets))
        SmoothL1(x) = 0.5 * (sigma * x)^2,    if |x| < 1 / sigma^2
                      |x| - 0.5 / sigma^2,    otherwise
    """
    sigma2 = sigma * sigma

    inside_mul = tf.multiply(bbox_inside_weights, tf.subtract(bbox_pred, bbox_targets))

    smooth_l1_sign = tf.cast(tf.less(tf.abs(inside_mul), 1.0 / sigma2), tf.float32)
    smooth_l1_option1 = tf.multiply(tf.multiply(inside_mul, inside_mul), 0.5 * sigma2)
    smooth_l1_option2 = tf.subtract(tf.abs(inside_mul), 0.5 / sigma2)
    smooth_l1_result = tf.add(tf.multiply(smooth_l1_option1, smooth_l1_sign),
                              tf.multiply(smooth_l1_option2, tf.abs(tf.subtract(smooth_l1_sign, 1.0))))

    outside_mul = tf.multiply(bbox_outside_weights, smooth_l1_result)

    return outside_mul


def do_reshape(input, d, name):
    input_shape = tf.shape(input)
    # transpose: (1, H, W, A x d) -> (1, H, WxA, d)
    return tf.reshape(input, \
                      [input_shape[0], \
                       input_shape[1], \
                       -1, \
                       int(d)])


def do_conv(x, layer_name, kernel_size, filter_size, stride_size, padding='SAME', activation_method=None, trainable=True, weights_initializer=tf.truncated_normal_initializer(0.0, stddev=0.01)):
    with tf.variable_scope(layer_name):
        in_shape = x.get_shape().as_list()
        # initializer = tf.contrib.layers.xavier_initializer()
        weights = get_weights('weights', shape=[kernel_size[0], kernel_size[1], in_shape[-1], filter_size],
                              initializer=weights_initializer, trainable=trainable)
        bias = get_weights('bias', shape=[filter_size], initializer=tf.constant_initializer(value=0.0), trainable=trainable)
        output = tf.nn.conv2d(x, filter=weights,
                              strides=[1, stride_size[0], stride_size[1], 1], padding=padding)
        output = tf.nn.bias_add(output, bias)
        if activation_method is not None:
            output = activation_method(output)
        return output


def do_maxpooling(x, layer_name, kernel_size, stride_size, padding='SAME'):
    with tf.variable_scope(layer_name):
        output = tf.nn.max_pool(x, ksize=[1, kernel_size[0], kernel_size[1], 1],
                                strides=[1, stride_size[0], stride_size[1], 1], padding=padding)
    return output


def do_upconv(x, layer_name, kernel_size, filter_size, output_shape, stride_size, padding, activation_method=None):
    with tf.variable_scope(layer_name):
        in_shape = x.get_shape().as_list()
        weights = get_weights('weights', shape=[kernel_size[0], kernel_size[1], filter_size, in_shape[-1]],
                              initializer=tf.contrib.layers.xavier_initializer())
        output = tf.nn.conv2d_transpose(x, weights, output_shape=output_shape,
                                        strides=[1, stride_size[0], stride_size[1], 1],
                                        padding=padding)
        bias = get_weights('bias', shape=[filter_size], initializer=tf.constant_initializer(value=0.0))
        output = tf.nn.bias_add(output, bias)
        if activation_method is not None:
            output = activation_method(output)
    return output


def do_fc(x, layer_name, output_node, relu=True, trainable=True):
    with tf.variable_scope(layer_name):
        if isinstance(x, tuple):
            x = x[0]
        input_shape = x.get_shape()
        if input_shape.ndims == 4:
            dim = 1
            for d in input_shape[1:].as_list():
                dim *= d
            feed_in = tf.reshape(x, [-1, dim])
        else:
            feed_in, dim = (x, int(input_shape[-1]))
        if layer_name == 'bbox_pred':
            init_weights = tf.truncated_normal_initializer(0.0, stddev=0.001)
            init_biases = tf.constant_initializer(0.0)
        else:
            init_weights = tf.truncated_normal_initializer(0.0, stddev=0.01)
            init_biases = tf.constant_initializer(0.0)
        weights = tf.get_variable('weights', [dim, output_node], initializer=init_weights, trainable=trainable,
                                  regularizer=l2_regularizer(cfg.TRAIN.WEIGHT_DECAY))
        bias = tf.get_variable('bias', [output_node], initializer=init_biases, trainable=trainable)

        op = tf.nn.relu_layer if relu else tf.nn.xw_plus_b
        fc = op(feed_in, weights, bias, name=layer_name)
        return fc

def l2_regularizer(weight_decay=0.0005, scope=None):
    def regularizer(tensor):
        with tf.name_scope(scope, default_name='l2_regularizer', values=[tensor]):
            l2_weight = tf.convert_to_tensor(weight_decay,
                                   dtype=tensor.dtype.base_dtype,
                                   name='weight_decay')
            #return tf.mul(l2_weight, tf.nn.l2_loss(tensor), name='value')
            return tf.multiply(l2_weight, tf.nn.l2_loss(tensor), name='value')
    return regularizer

def do_bilstm(input, layer_name, d_i, d_h, trainable=True):
    # d_i d_h d_o 512 128 512
    with tf.variable_scope(layer_name):
        shape = tf.shape(input)
        N, H, W, C = shape[0], shape[1], shape[2], shape[3]
        input = tf.reshape(input, [N * H, W, C])
        input.set_shape([None, None, d_i])
        fw_lstm = tf.nn.rnn_cell.LSTMCell(d_h, state_is_tuple=True)
        bw_lstm = tf.nn.rnn_cell.LSTMCell(d_h, state_is_tuple=True)
        outputs, output_states = tf.nn.bidirectional_dynamic_rnn(fw_lstm, bw_lstm, input, dtype=tf.float32)
        outputs = tf.concat(outputs, axis=-1)
        outputs = tf.reshape(outputs, [N*H*W, 2*d_h])
        return outputs


def do_bilstm_fc(input, layer_name, d_i, d_h, d_o, trainable=True):
    with tf.variable_scope(layer_name):
        shape = tf.shape(input)
        N, H, W, C = shape[0], shape[1], shape[2], shape[3]
        bilstm_outputs = do_bilstm(input, layer_name+'/bilstm', d_i, d_h, trainable)
        fc_outputs = do_fc(bilstm_outputs, layer_name+'/fc', d_o, relu=False, trainable=trainable)

        outputs = tf.reshape(fc_outputs, [N, H, W, d_o])
        print outputs
        return outputs


def lstm_fc(input, layer_name, d_i, d_o, trainable=True):
    with tf.variable_scope(layer_name):
        shape = tf.shape(input)
        N, H, W, C = shape[0], shape[1], shape[2], shape[3]
        input = tf.reshape(input, [N*H*W, d_i])

        weights = get_weights('weights', shape=[d_i, d_o], initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.01),
                              trainable=trainable)
        bias = get_weights('bias', [d_o], tf.constant_initializer(0.0), trainable=trainable)

        outputs = tf.nn.xw_plus_b(input, weights, bias)
        return tf.reshape(outputs, [N, H, W, d_o])


def load(data_path, session, saver, ignore_missing=False):
    if data_path.endswith('.ckpt'):
        saver.restore(session, data_path)
    else:
        data_dict = np.load(data_path).item()
        for key in data_dict:
            with tf.variable_scope(key, reuse=True):
                for subkey in data_dict[key]:
                    try:
                        var = tf.get_variable(subkey)
                        session.run(var.assign(data_dict[key][subkey]))
                        print "assign pretrain model "+subkey+ " to "+key
                    except ValueError:
                        print "ignore "+key
                        if not ignore_missing:
                            raise


def save(sess, saver, output_dir, prefix, infix, iter_index):

    filename = (prefix + infix +
                '_iter_{:d}'.format(iter_index + 1) + '.ckpt')
    filename = os.path.join(output_dir, filename)

    saver.save(sess, filename)

if __name__ == '__main__':
    input = tf.placeholder(dtype=tf.float32, shape=[None, 32, 32, 512])
    do_bilstm_fc(input, 'bilstm', 512, 128, 512)