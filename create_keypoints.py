import json
import os

import numpy as np
from PIL import Image
from matplotlib import pyplot as plt


def rubb_plot_fingers_orig(points, plt_specs, c, fig):
    """
    Plots the fingers of a single sample
    :param points: numpy array of shape (21, 3)
    :param plt_specs: the style of the plot # kuhn edited
    :param c: color
    :param ax: matplotlib axis
    """
    points = points[:, :2]
    for i in range(5):
        start, end = i * 4 + 1, (i + 1) * 4 + 1  # when i = 0, start = 1, end = 5 , when i = 1, start = 5, end = 9, when i = 2, start = 9, end = 13
        to_plot = np.concatenate((points[start:end], points[0:1]), axis=0)
        to_plot *= 1000
        fig.plot(to_plot[:, 0], to_plot[:, 1], plt_specs, color=c[i])  # draw the line
    # print("hello")


def read_keypoints(json_file_path):
    all_labels = json.load(json_file_path)
    img_labels = all_labels[img_name]  # origin label list  21 * 2
    # label = np.asarray(img_label)  # 21 * 2
    # return label
    # label_maps = gen_label_heatmap(label)
    return img_labels


if __name__ == '__main__':

    ################################################## param setting
    colorlist_gt = ['#000066', '#0000b3', '#0000ff', '#4d4dff', '#9999ff']
    data_path = r"low_low_light"
    label_file_path = r'images/CMUhand/labels.json'
    img_labels = json.load(open(label_file_path))
    ##################################################

    for img_name in os.listdir(data_path):
        img_path = os.path.join(data_path, img_name)
        plt.figure()
        img = Image.open(img_path)
        plt.imshow(img)
        keypoints = np.asarray(img_labels[img_name])  # 21 * 2
        rubb_plot_fingers_orig(keypoints, '.-', colorlist_gt, plt)
        plt.show()
        print("hello")
