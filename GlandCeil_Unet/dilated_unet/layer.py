import tensorflow as tf


# Weight initalization
def weight_xavier_init(shape, n_inputs, n_outputs, uniform=True, variable_name=None):
    if uniform:
        init_range = tf.sqrt(6.0 / (n_inputs + n_outputs))
        # 返回shape形状的矩阵，产生在（-init_range,init_range）之间，值均匀分布
        initial = tf.random_uniform(shape, -init_range, init_range)
        # 返回生成初始化随机权重矩阵
        return tf.Variable(initial, name=variable_name)
    else:
        stddev = tf.sqrt(3.0 / (n_inputs + n_outputs))
        initial = tf.truncated_normal(shape, stddev=stddev)
        return tf.Variable(initial, name=variable_name)


# Bias initialization
def bias_variable(shape, variable_name=None):
    # 创建常量
    initial = tf.constant(0.1, variable_name=None)
    return tf.Variable(initial, name=variable_name)


# 2D convolution
def conv2d(x, W, strides=1):
    # 卷积操作
    conv_2d = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    return conv_2d


def deconv2d(x, W, stride=2):
    # 获取输入维度
    x_shape = tf.shape(x)
    # 用于矩阵拼接
    output_shape = tf.stack([x_shape[0], x_shape[1] * stride, x_shape[2] * stride, x_shape[3] // stride])
    # 反卷积操作
    return tf.nn.conv2d_transpose(x, W, output_shape, strides=[1, stride, stride, 1], padding='SAME')


# 2D dilated Convolution
def dilated_conv2d(x, W, rate, padding):
    # 空洞卷积操作
    conv_2d = tf.nn.atrous_conv2d(x, W, rate, padding)
    return conv_2d


# Max Pooling
def max_pool_2x2(x):
    pool2d = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    return pool2d


# Unet crop and concat
def crop_and_concat(x1, x2):
    x1_shape = tf.shape(x1)
    x2_shape = tf.shape(x2)
    # offsets for the top left corner of the crop
    offsets = [0, (x1_shape[1] - x2_shape[1]) // 2, (x1_shape[2] - x2_shape[2]) // 2, 0]
    size = [-1, x2_shape[1], x2_shape[2], -1]
    x1_crop = tf.slice(x1, offsets, size)
    return tf.concat([x1_crop, x2], 3)
