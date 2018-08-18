"""
Using a convolutional neural network to crack captchas.

This approach is more complex than k-nearest neighbors,
but it's also much more general-purpose.
"""

import argparse
import os

from PIL import Image
from anyrl.utils.tf_state import load_vars, save_vars
import numpy as np
import tensorflow as tf

CAPTCHA_LENGTH = 5


def main():
    args = arg_parser().parse_args()
    train_images, train_labels = dataset_batch(args, dataset(args.data))
    test_images, test_labels = dataset_batch(args, dataset(args.test_data))

    with tf.variable_scope('model'):
        train_logits = model(train_images)
    with tf.variable_scope('model', reuse=True):
        test_logits = model(test_images)

    train_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=train_labels,
                                                                        logits=train_logits))
    test_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=test_labels,
                                                                       logits=test_logits))

    global_step = tf.get_variable('global_step',
                                  trainable=False,
                                  initializer=tf.constant(0, dtype=tf.int64))
    inc_step = tf.assign_add(global_step, tf.ones_like(global_step))

    with tf.control_dependencies([train_loss, test_loss, inc_step]):
        optim = tf.train.AdamOptimizer(learning_rate=args.lr).minimize(train_loss)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        load_vars(sess, args.model, var_list=tf.global_variables())
        while True:
            terms = sess.run((inc_step, train_loss, test_loss, optim))
            print('step %d: train=%f test=%f' % terms[:-1])
            if terms[0] % args.save_interval == 0:
                save_vars(sess, args.model, var_list=tf.global_variables())


def model(images):
    """
    Apply a CNN model to the captcha images to get logits.

    Args:
      images: a uint8 [batch x 50 x 200 x 3] Tensor.

    Returns:
      A logit Tensor of shape [batch x num_digits x 10].
    """
    out = tf.cast(images, tf.float32) / 127.5 - 1
    out = tf.layers.conv2d(out, 16, 3, strides=2, padding='same', activation=tf.nn.relu)
    out = tf.layers.conv2d(out, 16, 3, strides=2, padding='same', activation=tf.nn.relu)
    out = tf.layers.conv2d(out, 16, 3, strides=2, padding='same', activation=tf.nn.relu)
    # Shape is now [batch x 7 x (5 * CAPTCHA_LENGTH) x 16]

    # Slide a convolution horizontally and use each patch
    # to produce some logits.
    patch_size = (out.get_shape()[1], out.get_shape()[2] // CAPTCHA_LENGTH)
    out = tf.layers.conv2d(out, 10, kernel_size=patch_size, strides=patch_size, padding='valid')
    return tf.reshape(out, [-1, CAPTCHA_LENGTH, 10])


def dataset(data_dir):
    """
    Create a tf.data.Dataset of (image, label) pairs,
    where each label is a [num_digits x 10] sparse matrix.
    """
    arrays = []
    labels = []
    for file in os.listdir(data_dir):
        if not file.endswith('.jpg'):
            continue
        arrays.append(np.array(Image.open(os.path.join(data_dir, file))))
        digit_labels = []
        for digit in file[:CAPTCHA_LENGTH]:
            one_hot = [0.0] * 10
            one_hot[int(digit)] = 1.0
            digit_labels.append(one_hot)
        labels.append(digit_labels)
    return tf.data.Dataset.from_tensor_slices((np.array(arrays), np.array(labels, dtype='float32')))


def dataset_batch(args, dataset):
    """
    Create a mini-batch for the tf.data.Dataset.
    """
    dataset = dataset.shuffle(10000)
    dataset = dataset.batch(args.batch)
    dataset = dataset.repeat()
    return dataset.make_one_shot_iterator().get_next()


def arg_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--model', help='model save path', default='cnn_model.pkl')
    parser.add_argument('--save-interval', help='steps per save', default=10, type=int)

    parser.add_argument('--data', help='training images', default='data/auto')
    parser.add_argument('--test-data', help='testing images', default='data/test')

    parser.add_argument('--batch', help='batch size', default=16, type=int)
    parser.add_argument('--lr', help='Adam step size', default=0.001, type=float)

    return parser


if __name__ == '__main__':
    main()
