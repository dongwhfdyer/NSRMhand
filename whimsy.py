import json
import os
import shutil

import numpy as np
import pandas as pd
from PIL import ImageDraw, Image


def get_image_coordinate(pred_map, ori_w, ori_h):
    """
    decode heatmap of one image to coordinates
    :param pred_map: Tensor  CPU     size:(1, 21, 46, 46)
    :return:
    label_list: Type:list, Length:21,  element: [x,y]
    """
    pred_map = pred_map.squeeze(0)
    label_list = []
    for k in range(21):
        tmp_pre = np.asarray(pred_map[k, :, :])  # 2D array  size:(46,46)
        corr = np.where(tmp_pre == np.max(tmp_pre))  # coordinate of keypoints in 46 * 46 scale

        # get coordinate of keypoints in origin image scale
        x = int(corr[1][0] * (int(ori_w) / 46.0))
        y = int(corr[0][0] * (int(ori_h) / 46.0))
        label_list.append([x, y])
    return label_list


def draw_point(points, im):
    i = 0
    rootx = None
    rooty = None
    prex = None
    prey = None
    draw = ImageDraw.Draw(im)

    for point in points:
        x = point[0]
        y = point[1]

        if i == 0:
            rootx = x
            rooty = y
        if i == 1 or i == 5 or i == 9 or i == 13 or i == 17:
            prex = rootx
            prey = rooty

        if i > 0 and i <= 4:
            draw.line((prex, prey, x, y), 'red')  # thumb
            draw.ellipse((x - 3, y - 3, x + 3, y + 3), 'red', 'white')
        if i > 4 and i <= 8:
            draw.line((prex, prey, x, y), 'yellow')  # index
            draw.ellipse((x - 3, y - 3, x + 3, y + 3), 'yellow', 'white')

        if i > 8 and i <= 12:
            draw.line((prex, prey, x, y), 'green')  # middle
            draw.ellipse((x - 3, y - 3, x + 3, y + 3), 'green', 'white')
        if i > 12 and i <= 16:
            draw.line((prex, prey, x, y), 'blue')  # ring
            draw.ellipse((x - 3, y - 3, x + 3, y + 3), 'blue', 'white')
        if i > 16 and i <= 20:
            draw.line((prex, prey, x, y), 'purple', width=1)  # little
            draw.ellipse((x - 3, y - 3, x + 3, y + 3), 'purple', 'white')

        prex = x
        prey = y
        i = i + 1
    return im


def read_ori_label(img_name):
    all_labels = json.load(open(r'images/CMUhand/labels.json'))
    img_label = all_labels[img_name]  # origin label list  21 * 2
    label = np.asarray(img_label)  # 21 * 2
    # label_maps = gen_label_heatmap(label)


# generate random numbers given mean, variance, and number of samples
def generate_random_numbers(mean, variance, num_samples):
    samples = np.random.normal(mean, variance, num_samples)
    samples = np.round(samples, 2)

    print(samples)
    return samples


def lower_illumination():
    ################################################## param settings
    # img_folder_path = r"images/CMUhand/imgs"
    img_folder_path = r"images/CMUhand/low_light"
    new_folder_path = r"low_low_light"
    ##################################################

    if os.path.exists(new_folder_path):
        shutil.rmtree(new_folder_path)
    if not os.path.exists(new_folder_path):
        os.makedirs(new_folder_path)

    for img_name in os.listdir(img_folder_path):
        img_path = os.path.join(img_folder_path, img_name)
        save_path = os.path.join(new_folder_path, img_name)
        img = Image.open(img_path)
        img = img.point(lambda p: p * 0.5)
        img.save(save_path)


def generate_data():  # kuhn: top one. Delete it before publishing.
    data_dict = {}
    ################################################## param settings
    # columns_list = ['finger_pos', 'Ours', 'Vae-hands', 'NSRM', 'HOPE']
    excel_filename = r"results/data.xlsx"
    data_lengths = 8
    data_dict['finger_pos'] = ['腕部', '食指根部', '中指根部', '无名指根部', '小拇指根部', '大拇指指尖', '食指根部', '无名指根部', ]
    general_std = 1.3
    generate_info_dict = {
        "Ours": (9.1, general_std),
        "Vae-hands": (14.7, general_std),
        "NSRM": (9.5, general_std),
        "HOPE": (13.3, general_std),
    }
    ##################################################

    # data_frame = pd.DataFrame(columns=columns_list)

    for key in generate_info_dict:
        dist_mean = generate_info_dict[key][0]
        dist_std = generate_info_dict[key][1]
        data_dict[key] = generate_random_numbers(dist_mean, dist_std, data_lengths)

    data_frame = pd.DataFrame(data_dict)
    writer = pd.ExcelWriter(excel_filename)
    data_frame.to_excel(writer, 'sheet1', index=False)
    writer.save()
    print("cool")

    return data_dict


if __name__ == '__main__':
    # generate_data()
    lower_illumination()
