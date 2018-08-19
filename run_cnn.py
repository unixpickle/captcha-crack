"""
Run the CNN on a batch of images.
"""

import argparse
import sys

from PIL import Image
from anyrl.utils.tf_state import load_vars
import numpy as np
import tensorflow as tf

from cnn import model


def main():
    args = arg_parser().parse_args()

    image_ph = tf.placeholder(tf.uint8, shape=(50, 200, 3))
    with tf.variable_scope('model'):
        logits = model(image_ph[None])
    labels = tf.argmax(logits, axis=-1)[0]

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        load_vars(sess, args.model, relaxed=True, log_fn=load_log_fn)
        for image_path in args.images:
            image = np.array(Image.open(image_path))
            labels_out = sess.run(labels, feed_dict={image_ph: image})
            print('%s: %s' % (image_path, ''.join(map(str, labels_out))))


def load_log_fn(msg):
    if not any([x in msg for x in ['Adam', 'global_step', 'beta1', 'beta2']]):
        sys.stderr.write(msg + '\n')


def arg_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', help='model save path', default='cnn_model.pkl')
    parser.add_argument('images', help='image files', nargs='+')
    return parser


if __name__ == '__main__':
    main()
