from GlandCeil_Unet.dilated_unet.layer import (dilated_conv2d, conv2d, deconv2d, max_pool_2x2, crop_and_concat,
                                               weight_xavier_init, bias_variable)
import tensorflow as tf
import numpy as np
import os
import cv2


def _create_dilated_conv_net(X, image_width, image_height, image_channel, phase, drop_conv, n_class=1):
    # 输入形状
    inputX = tf.reshape(X, [-1, image_width, image_height, image_channel])
    # dilated U-Net
    # layer1->convolution
    W1_1 = weight_xavier_init(shape=[3, 3, image_channel, 32], n_inputs=3 * 3 * image_channel, n_outputs=32)
    B1_1 = bias_variable([32])
    conv1_1 = conv2d(inputX, W1_1) + B1_1
