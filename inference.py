import argparse
import json
import os
import shutil

import torch
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms
from tqdm import tqdm

from model.cpm_limb import CPMHandLimb
from PIL import Image, ImageDraw

cuda = torch.cuda.is_available()
device_id = [0]
torch.cuda.set_device(device_id[0])


def load_image(img_path):
    ori_im = Image.open(img_path)
    ori_w, ori_h = ori_im.size
    im = ori_im.resize((368, 368))
    image = transforms.ToTensor()(im)
    image = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])(image)  # (C,H,W)
    image = image.unsqueeze(0)  # (1,C,H,W)
    return ori_im, image, ori_w, ori_h


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


def generate_abnormal_coordinates(coordinates_abnormal):
    mutilate_num = [1, 2, ]
    # mutilate_num = [1, 2, 3]
    fingers = [0, 1, 2, ]
    # fingers = [0, 1, 2, 3, 4, ]
    bias = [4, 8, ]
    # bias = [4, 8, 12,]

    for i in range(len(coordinates_abnormal)):  # generate three abnormal images
        mutilate_num_choice = np.random.choice(mutilate_num)
        fingers_choice = np.random.choice(fingers, mutilate_num_choice, replace=False)
        bias_choice = np.random.choice(bias, mutilate_num_choice, replace=False)
        for j in range(mutilate_num_choice):
            current_finger_index = fingers_choice[j]
            current_bias = bias_choice[j]
            start_ind = 1 + current_finger_index * 4
            end_ind = start_ind + 4
            coordinates_abnormal[i][start_ind:end_ind] = coordinates_abnormal[i][start_ind:end_ind] + [current_bias, current_bias]
            # print("hello")
    return coordinates_abnormal


def hand_pose_estimation(model, img_path='images/sample.jpg', save_path='images/sample_out.jpg'):
    with torch.no_grad():
        ori_imm, img, ori_w, ori_h = load_image(img_path)  # ori_im: PIL.Image.Image, img: tensor, ori_w, ori_h: int
        # img is for inference
        if cuda:
            img = img.cuda()  # # Tensor size:(1,3,368,368)
        _, cm_pred = model(img)
        # limb_pred (FloatTensor.cuda) size:(bz,3,C,46,46)
        # cm_pred   (FloatTensor.cuda) size:(bz,3,21,46,46)

        ori_coordinates = read_ori_label(img_path[-12:])
        ################################################## wonder
        # get three copy of origin coordinates
        coordinates_abnormal = [ori_coordinates.copy(), ori_coordinates.copy(), ori_coordinates.copy(), ori_coordinates.copy()]
        # for i in range(3):
        #     print(id(coordinates_abnormal[i]))

        coordinates_abnormal = generate_abnormal_coordinates(coordinates_abnormal)
        ori_imm.save(save_path)
        ori_im = draw_point(ori_coordinates, ori_imm)
        save_path_ = save_path[:-4] + '_aadd_pos.jpg'
        ori_im.save(save_path_)

        drawing_list = []
        for i in range(4):
            drawing_list.append(draw_point(coordinates_abnormal[i], ori_imm))
            save_path_ = save_path.replace('.jpg', '_abnormal_' + str(i) + '.jpg')
            # print('save output to ', save_path_)
            drawing_list[i].save(save_path_)
        # print("id", id(drawing_list[i]))

        ##################################################
        # coordinates = get_image_coordinate(cm_pred[:, -1].cpu(), ori_w, ori_h)
        # ori_im = draw_point(coordinates, ori_im)
        # show image


def read_ori_label(img_name):
    all_labels = json.load(open(r'images/CMUhand/labels.json'))
    img_label = all_labels[img_name]  # origin label list  21 * 2
    label = np.asarray(img_label)  # 21 * 2
    return label
    # label_maps = gen_label_heatmap(label)


def draw_point(points, imm):
    im = imm.copy()
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

        dot_size = 1.3
        if i > 0 and i <= 4:
            draw.line((prex, prey, x, y), 'red')  # thumb
            draw.ellipse((x - dot_size, y - dot_size, x + dot_size, y + dot_size), 'red', 'white')
        if i > 4 and i <= 8:
            draw.line((prex, prey, x, y), 'yellow')  # index
            draw.ellipse((x - dot_size, y - dot_size, x + dot_size, y + dot_size), 'yellow', 'white')

        if i > 8 and i <= 12:
            draw.line((prex, prey, x, y), 'green')  # middle
            draw.ellipse((x - dot_size, y - dot_size, x + dot_size, y + dot_size), 'green', 'white')
        if i > 12 and i <= 16:
            draw.line((prex, prey, x, y), 'blue')  # ring
            draw.ellipse((x - dot_size, y - dot_size, x + dot_size, y + dot_size), 'blue', 'white')
        if i > 16 and i <= 20:
            draw.line((prex, prey, x, y), 'purple', width=0)  # little
            draw.ellipse((x - dot_size, y - dot_size, x + dot_size, y + dot_size), 'purple', 'white')

        prex = x
        prey = y
        i = i + 1
    return im.copy()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', default='weights/best_model.pth', help='trained model dir')
    parser.add_argument('--image_dir', default='images/', help='path for folder')
    parser.add_argument('--save_dir', default='results/run_1', help='path for folder')
    args = parser.parse_args()

    ################################################## param setting
    # args.image_dir = r"low_low_light"
    args.image_dir = r"images/CMUhand/low_light"
    # args.image_dir = r"images/CMUhand/imgs"
    args.save_dir = r"results/run_2"
    ##################################################

    if os.path.exists(args.save_dir):
        shutil.rmtree(args.save_dir)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # Limb Probabilistic Mask G1 & 6
    model = CPMHandLimb(outc=21, lshc=7, pretrained=False)
    if cuda:
        model = model.cuda()
        model = nn.DataParallel(model, device_id)

    state_dict = torch.load(args.resume, map_location='cuda:0')
    model.load_state_dict(state_dict)

    ################################################## inference
    images_path = os.listdir(args.image_dir)
    pbar = tqdm(total=len(images_path))
    for image_name in images_path:
        pbar.update(1)
        image_path = os.path.join(args.image_dir, image_name)
        save_path = os.path.join(args.save_dir, image_name)
        hand_pose_estimation(model, image_path, save_path)
    ##################################################
