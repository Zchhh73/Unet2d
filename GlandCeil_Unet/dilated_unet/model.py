from GlandCeil_Unet.dilated_unet.layer import (dilated_conv2d, conv2d, deconv2d, max_pool_2x2, crop_and_concat,
                                               weight_xavier_init, bias_variable)
import tensorflow as tf
import numpy as np
import os
import cv2


def get_dilated_unet(X, image_width, image_height, image_channel, phase, drop_conv, n_class=1, dilaterate=3):
    inputX = tf.reshape(X, [-1, image_width, image_height, image_channel])
    # Dilated_UNet model
    # layer1->convolution
    W1_1 = weight_xavier_init(shape=[3, 3, image_channel, 32], n_inputs=3 * 3 * image_channel, n_outputs=32)
    B1_1 = bias_variable([32])
    conv1_1 = conv2d(inputX, W1_1) + B1_1
    conv1_1 = tf.contrib.layers.batch_norm(conv1_1, center=True, scale=True, is_training=phase, scope='bn1')
    conv1_1 = tf.nn.dropout(tf.nn.relu(conv1_1), drop_conv)

    W1_2 = weight_xavier_init(shape=[3, 3, 32, 32], n_inputs=3 * 3 * 32, n_outputs=32)
    B1_2 = bias_variable([32])
    conv1_2 = conv2d(conv1_1, W1_2) + B1_2
    # conv1_2 = tf.contrib.layers.batch_norm(conv1_2, epsilon=1e-5, scope='bn2')
    conv1_2 = tf.contrib.layers.batch_norm(conv1_2, center=True, scale=True, is_training=phase, scope='bn2')
    conv1_2 = tf.nn.dropout(tf.nn.relu(conv1_2), drop_conv)
    pool1 = max_pool_2x2(conv1_2)

    # layer2->convolution
    W2_1 = weight_xavier_init(shape=[3, 3, 32, 64], n_inputs=3 * 3 * 32, n_outputs=64)
    B2_1 = bias_variable([64])
    conv2_1 = conv2d(pool1, W2_1) + B2_1
    conv2_1 = tf.contrib.layers.batch_norm(conv2_1, center=True, scale=True, is_training=phase, scope='bn3')
    conv2_1 = tf.nn.dropout(tf.nn.relu(conv2_1), drop_conv)

    W2_2 = weight_xavier_init(shape=[3, 3, 64, 64], n_inputs=3 * 3 * 64, n_outputs=64)
    B2_2 = bias_variable([64])
    conv2_2 = conv2d(conv2_1, W2_2) + B2_2
    conv2_2 = tf.contrib.layers.batch_norm(conv2_2, center=True, scale=True, is_training=phase, scope='bn4')
    conv2_2 = tf.nn.dropout(tf.nn.relu(conv2_2), drop_conv)
    pool2 = max_pool_2x2(conv2_2)

    # layer3->convolution
    W3_1 = weight_xavier_init(shape=[3, 3, 64, 128], n_inputs=3 * 3 * 64, n_outputs=128)
    B3_1 = bias_variable([128])
    conv3_1 = dilated_conv2d(pool2, W3_1, 2, 'SAME') + B3_1
    conv3_1 = tf.contrib.layers.batch_norm(conv3_1, center=True, scale=True, is_training=phase, scope='bn5')
    conv3_1 = tf.nn.dropout(tf.nn.relu(conv3_1), drop_conv)

    W3_2 = weight_xavier_init(shape=[3, 3, 128, 128], n_inputs=3 * 3 * 128, n_outputs=128)
    B3_2 = bias_variable([128])
    conv3_2 = dilated_conv2d(conv3_1, W3_2, 2, 'SAME') + B3_2
    conv3_2 = tf.contrib.layers.batch_norm(conv3_2, center=True, scale=True, is_training=phase, scope='bn6')
    conv3_2 = tf.nn.dropout(tf.nn.relu(conv3_2), drop_conv)
    pool3 = max_pool_2x2(conv3_2)

    # layer4->Convolution
    W4_1 = weight_xavier_init(shape=[3, 3, 128, 256], n_inputs=3 * 3 * 128, n_outputs=256)
    B4_1 = bias_variable([256])
    conv4_1 = dilated_conv2d(pool3, W4_1, 2, 'SAME') + B4_1
    conv4_1 = tf.contrib.layers.batch_norm(conv4_1, center=True, scale=True, is_training=phase, scope='bn7')
    conv4_1 = tf.nn.dropout(tf.nn.relu(conv4_1), drop_conv)

    W4_2 = weight_xavier_init(shape=[3, 3, 256, 256], n_inputs=3 * 3 * 256, n_outputs=256)
    B4_2 = bias_variable([256])
    conv4_2 = dilated_conv2d(conv4_1, W4_2, 2, 'SAME') + B4_2
    conv4_2 = tf.contrib.layers.batch_norm(conv4_2, center=True, scale=True, is_training=phase, scope='bn8')
    conv4_2 = tf.nn.dropout(tf.nn.relu(conv4_2), drop_conv)

    # layer5->deConvolution
    W5 = weight_xavier_init(shape=[3, 3, 128, 256], n_inputs=3 * 3 * 256, n_outputs=128)
    B5 = bias_variable([128])
    dconv1 = tf.nn.relu(deconv2d(conv4_2, W5) + B5)
    dconv_concat1 = crop_and_concat(conv3_2, dconv1)

    # layer6->Convolution
    W6_1 = weight_xavier_init(shape=[3, 3, 256, 128], n_inputs=3 * 3 * 256, n_outputs=128)
    B6_1 = bias_variable([128])
    conv6_1 = conv2d(dconv_concat1, W6_1) + B6_1
    conv6_1 = tf.contrib.layers.batch_norm(conv6_1, center=True, scale=True, is_training=phase, scope='bn11')
    conv6_1 = tf.nn.dropout(tf.nn.relu(conv6_1), drop_conv)

    W6_2 = weight_xavier_init(shape=[3, 3, 128, 128], n_inputs=3 * 3 * 128, n_outputs=128)
    B6_2 = bias_variable([128])
    conv6_2 = conv2d(conv6_1, W6_2) + B6_2
    conv6_2 = tf.contrib.layers.batch_norm(conv6_2, center=True, scale=True, is_training=phase, scope='bn12')
    conv6_2 = tf.nn.dropout(tf.nn.relu(conv6_2), drop_conv)

    # layer7->deConvolution
    W7 = weight_xavier_init(shape=[3, 3, 64, 128], n_inputs=3 * 3 * 128, n_outputs=64)
    B7 = bias_variable([64])
    dconv2 = tf.nn.relu(deconv2d(conv6_2, W7) + B7)
    dconv_concat2 = crop_and_concat(conv2_2, dconv2)

    # layer8->Convolution
    W8_1 = weight_xavier_init(shape=[3, 3, 128, 64], n_inputs=3 * 3 * 128, n_outputs=64)
    B8_1 = bias_variable([64])
    conv8_1 = conv2d(dconv_concat2, W8_1) + B8_1
    conv8_1 = tf.contrib.layers.batch_norm(conv8_1, center=True, scale=True, is_training=phase, scope='bn11')
    conv8_1 = tf.nn.dropout(tf.nn.relu(conv8_1), drop_conv)

    W8_2 = weight_xavier_init(shape=[3, 3, 64, 64], n_inputs=3 * 3 * 64, n_outputs=64)
    B8_2 = bias_variable([64])
    conv8_2 = conv2d(conv6_1, W8_2) + B8_2
    conv8_2 = tf.contrib.layers.batch_norm(conv8_2, center=True, scale=True, is_training=phase, scope='bn12')
    conv8_2 = tf.nn.dropout(tf.nn.relu(conv8_2), drop_conv)

    # layer9->deConvolution
    W9 = weight_xavier_init(shape=[3, 3, 32, 64], n_inputs=3 * 3 * 64, n_outputs=32)
    B9 = bias_variable([32])
    dconv2 = tf.nn.relu(deconv2d(conv8_2, W9) + B9)
    dconv_concat2 = crop_and_concat(conv1_2, dconv2)

    # layer10->Convolution
    W10_1 = weight_xavier_init(shape=[3, 3, 64, 32], n_inputs=3 * 3 * 64, n_outputs=32)
    B10_1 = bias_variable([32])
    conv10_1 = conv2d(dconv_concat2, W10_1) + B10_1
    conv10_1 = tf.contrib.layers.batch_norm(conv10_1, center=True, scale=True, is_training=phase, scope='bn11')
    conv10_1 = tf.nn.dropout(tf.nn.relu(conv10_1), drop_conv)

    W10_2 = weight_xavier_init(shape=[3, 3, 32, 32], n_inputs=3 * 3 * 32, n_outputs=32)
    B10_2 = bias_variable([32])
    conv10_2 = conv2d(conv10_1, W10_2) + B10_2
    conv10_2 = tf.contrib.layers.batch_norm(conv10_2, center=True, scale=True, is_training=phase, scope='bn12')
    conv10_2 = tf.nn.dropout(tf.nn.relu(conv10_2), drop_conv)

    # layer11->output
    W11 = weight_xavier_init(shape=[1, 1, 32, n_class], n_inputs=1 * 1 * 32, n_outputs=n_class)
    B11 = bias_variable([n_class])
    output_map = tf.nn.sigmoid(conv2d(conv10_2, W11) + B11, name='output')
    return output_map


def _next_batch(train_images, train_labels, batch_size, index_in_epoch):
    start = index_in_epoch
    index_in_epoch += batch_size

    num_examples = train_images.shape[0]
    # when all trainig data have been already used, it is reorder randomly
    if index_in_epoch > num_examples:
        # shuffle the data
        perm = np.arange(num_examples)
        np.random.shuffle(perm)
        train_images = train_images[perm]
        train_labels = train_labels[perm]
        # start next epoch
        start = 0
        index_in_epoch = batch_size
        assert batch_size <= num_examples
    end = index_in_epoch
    return train_images[start:end], train_labels[start:end], index_in_epoch


class dilatedUnet2dModule(object):
    def __init__(self, image_height, image_width, channels=1, costname="dice coefficient"):
        self.image_with = image_width
        self.image_height = image_height
        self.channels = channels

        self.X = tf.placeholder("float", shape=[None, image_height, image_width, channels], name="Input")
        self.Y_gt = tf.placeholder("float", shape=[None, image_height, image_width, 1], name="Output_GT")
        self.lr = tf.placeholder('float', name="Learning_rate")
        self.phase = tf.placeholder(tf.bool, name="Phase")
        self.drop_conv = tf.placeholder('float', name="DropOut")

        self.Y_pred = _create_conv_net(self.X, image_width, image_height, channels, self.phase, self.drop_conv)

        self.cost = self.__get_cost(costname)
        self.accuracy = -self.__get_cost(costname)

    def __get_cost(self, cost_name):
        H, W, C = self.Y_gt.get_shape().as_list()[1:]
        if cost_name == "dice coefficient":
            smooth = 1e-5
            pred_flat = tf.reshape(self.Y_pred, [-1, H * W * C])
            true_flat = tf.reshape(self.Y_gt, [-1, H * W * C])
            intersection = 2 * tf.reduce_sum(pred_flat * true_flat, axis=1) + smooth
            denominator = tf.reduce_sum(pred_flat, axis=1) + tf.reduce_sum(true_flat, axis=1) + smooth
            loss = -tf.reduce_mean(intersection / denominator)
        if cost_name == "pixelwise_cross entroy":
            assert (C == 1)
            flat_logit = tf.reshape(self.Y_pred, [-1])
            flat_label = tf.reshape(self.Y_gt, [-1])
            loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=flat_logit, labels=flat_label))
        return loss

    def train(self, train_images, train_labels, model_path, logs_path, learning_rate,
              dropout_conv=0.8, train_epochs=1000, batch_size=2):
        train_op = tf.train.AdamOptimizer(self.lr).minimize(self.cost)
        init = tf.global_variables_initializer()
        saver = tf.train.Saver(tf.all_variables())

        tf.summary.scalar("loss", self.cost)
        tf.summary.scalar("accuracy", self.accuracy)
        merged_summary_op = tf.summary.merge_all()
        sess = tf.InteractiveSession(config=tf.ConfigProto(allow_soft_placement=True))
        summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())
        sess.run(init)
        if os.path.isfile(model_path):
            saver.restore(sess, model_path)

        DISPLAY_STEP = 1
        index_in_epoch = 0

        for i in range(train_epochs):
            batch_xs_path, batch_ys_path, index_in_epoch = _next_batch(train_images, train_labels, batch_size,
                                                                       index_in_epoch)
            batch_xs = np.empty((len(batch_xs_path), self.image_height, self.image_with, self.channels))
            batch_ys = np.empty((len(batch_ys_path), self.image_height, self.image_with, 1))

            for num in range(len(batch_xs_path)):
                image = cv2.imread(batch_xs_path[num][0], cv2.IMREAD_COLOR)
                # cv2.imwrite('image_src.bmp', image)
                label = cv2.imread(batch_ys_path[num][0], cv2.IMREAD_GRAYSCALE)
                # cv2.imwrite('mask.bmp', label)
                batch_xs[num, :, :, :] = np.reshape(image, (self.image_height, self.image_with, self.channels))
                batch_ys[num, :, :, :] = np.reshape(label, (self.image_height, self.image_with, 1))
            # Extracting images and labels from given data
            batch_xs = batch_xs.astype(np.float)
            batch_ys = batch_ys.astype(np.float)
            # Normalize from [0:255] => [0.0:1.0]
            batch_xs = np.multiply(batch_xs, 1.0 / 255.0)
            batch_ys = np.multiply(batch_ys, 1.0 / 255.0)
            # check progress on every 1st,2nd,...,10th,20th,...,100th... step
            if i % DISPLAY_STEP == 0 or (i + 1) == train_epochs:
                train_loss, train_accuracy = sess.run([self.cost, self.accuracy], feed_dict={self.X: batch_xs,
                                                                                             self.Y_gt: batch_ys,
                                                                                             self.lr: learning_rate,
                                                                                             self.phase: 1,
                                                                                             self.drop_conv: dropout_conv})
                pred = sess.run(self.Y_pred, feed_dict={self.X: batch_xs,
                                                        self.Y_gt: batch_ys,
                                                        self.phase: 1,
                                                        self.drop_conv: 1})
                result = np.reshape(pred[0], (512, 512))
                result = result.astype(np.float32) * 255.
                result = np.clip(result, 0, 255).astype('uint8')
                cv2.imwrite("result.bmp", result)
                print('epochs %d training_loss ,Training_accuracy => %.5f,%.5f ' % (i, train_loss, train_accuracy))
                if i % (DISPLAY_STEP * 10) == 0 and i:
                    DISPLAY_STEP *= 10

                    # train on batch
            _, summary = sess.run([train_op, merged_summary_op], feed_dict={self.X: batch_xs,
                                                                            self.Y_gt: batch_ys,
                                                                            self.lr: learning_rate,
                                                                            self.phase: 1,
                                                                            self.drop_conv: dropout_conv})
            summary_writer.add_summary(summary, i)
        summary_writer.close()

        save_path = saver.save(sess, model_path)
        print("Model saved in file:", save_path)

    def prediction(self, model_path, test_images):
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()
        sess = tf.InteractiveSession()
        sess.run(init)
        saver.restore(sess, model_path)

        test_images = np.reshape(test_images, (1, test_images.shape[0], test_images.shape[1], self.channels))
        pred = sess.run(self.Y_pred, feed_dict={self.X: test_images,
                                                self.phase: 1,
                                                self.drop_conv: 1})
        result = np.reshape(pred, (test_images.shape[1], test_images.shape[2]))
        result = result.astype(np.float32) * 255.
        result = np.clip(result, 0, 255).astype('uint8')
        return result
