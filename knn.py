"""
Classify captchas using per-digit k-nearest neighbors.
"""

import argparse
from collections import Counter
import os

from PIL import Image
import numpy as np

CAPTCHA_LENGTH = 5


def main():
    args = arg_parser().parse_args()

    print('Loading dataset...')
    training_data = load_data(args.data)

    print('Classifying test set...')
    total_images = 0
    correct_images = 0
    for image, label in images_in_dir(args.test_data):
        total_images += 1
        if classify_image(training_data, image) == label:
            correct_images += 1
    print('Got %d/%d' % (correct_images, total_images))

    print('Auto-renaming images...')
    for image, name in images_in_dir(args.auto_data, full_name=True):
        out_name = classify_image(training_data, image) + '.jpg'
        print('%s -> %s' % (name, out_name))
        os.rename(os.path.join(args.auto_data, name),
                  os.path.join(args.auto_data, out_name))


def load_data(data_dir):
    """
    Load the labeled digit images from a directory.

    Returns:
      A tuple (images, labels):
        images: an [N x height x width x 3] array of uint8
          pixel values.
        labels: an N-dimensional array of integer labels.
    """
    images = []
    labels = []
    for image, label in images_in_dir(data_dir):
        cell_width = image.shape[1] // len(label)
        for i, digit in enumerate(label):
            images.append(image[:, cell_width * i:cell_width * (i + 1)])
            labels.append(int(digit))
    return np.array(images), np.array(labels)


def classify_image(training_data, image, num_neighbors=3):
    """
    Label a captcha image using the training data.

    Args:
      training_data: the data returned by load_data().
      image: a [height x width x 3] captcha image.
      num_neighbors: "k" in k-nearest neighbors.

    Returns:
      A string of captcha text.
    """
    images, labels = training_data
    images = images.astype('float') / 255
    image = image.astype('float') / 255
    cell_width = images.shape[2]
    result = []
    for i in range(0, image.shape[1], cell_width):
        patch = image[:, i:i + cell_width]
        distances = np.sum(np.square(images - patch), axis=(1, 2, 3))
        neighbor_labels = labels[np.argsort(distances)][:num_neighbors]
        majority = sorted(Counter(neighbor_labels).items(), key=lambda x: x[1])[-1][0]
        result.append(majority)
    return ''.join(map(str, result))


def images_in_dir(data_dir, full_name=False):
    for file in os.listdir(data_dir):
        if not file.endswith('.jpg'):
            continue
        img = np.array(Image.open(os.path.join(data_dir, file)))
        yield img, (file if full_name else file[:CAPTCHA_LENGTH])


def arg_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data', help='training images', default='data/train')
    parser.add_argument('--test-data', help='testing images', default='data/test')
    parser.add_argument('--auto-data', help='images to rename with labels', default='data/auto')
    return parser


if __name__ == '__main__':
    main()
