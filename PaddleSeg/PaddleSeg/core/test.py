# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import math
from typing import List

import cv2
import numpy as np
import paddle
from paddle.fluid.layers.tensor import save
from PIL import Image
import multiprocessing
from tqdm import tqdm

from paddleseg import utils
from paddleseg.core import infer
from paddleseg.utils import logger, progbar
from paddleseg.transforms import Compose

from paddle.io import Dataset, DataLoader


rgb = {
    0: [255,0,0],
    1: [0,255,0],
    2: [0,0,255],
    3: [255,255,255],
    255: [0,0,0]
}


def mkdir(path):
    sub_dir = os.path.dirname(path)
    if not os.path.exists(sub_dir):
        os.makedirs(sub_dir)


def gray2rgb(data):
    h,w = data.shape[:2]
    rgb_data = np.zeros([h,w,3], dtype=np.uint8)
    for i in range(4):
        rgb_data[data==i] = rgb[i]
    return rgb_data


class My_dataset(Dataset):
    def __init__(self, file_path, mode='val', transform=[]):
        self.file_list = list()
        self.transforms = Compose(transform)
        self.mode = mode
        with open(file_path, 'r') as f:
            for line in f:
                items = line.strip().split(' ')
                if len(items) != 2:
                    image_path = items[0]
                    label_path = 0
                else:
                    image_path = items[0]
                    label_path = items[1]
                self.file_list.append([image_path, label_path])
    
    def __getitem__(self, idx):
        image_path, label_path = self.file_list[idx]
        # name = os.path.basename(image_path).split('.')[0]
        if self.mode == 'test':
            im, _ = self.transforms(im=image_path) # img:[3,256,256]
            return im, 0, image_path
        elif self.mode == 'val':
            im, label = self.transforms(im=image_path, label=label_path)
            label = gray2rgb(label)
            return im, label, image_path  # img:[3,256,256] label:[256,256,3]

    def __len__(self):
        return len(self.file_list)


def test(model,
        model_path,
        transforms,
        file_path,
        save_dir='output',
        mode_forinfer='test',
        aug_pred=False,
        scales=1.0,
        flip_horizontal=True,
        flip_vertical=False,
        is_slide=False,
        stride=None,
        crop_size=None):
    """
    predict and visualize the image_list.

    Args:
        model (nn.Layer): Used to predict for input image.
        model_path (str): The path of pretrained model.
        transforms (transform.Compose): Preprocess for input image.
        image_list (list): A list of image path to be predicted.
        image_dir (str, optional): The root directory of the images predicted. Default: None.
        save_dir (str, optional): The directory to save the visualized results. Default: 'output'.
        aug_pred (bool, optional): Whether to use mulit-scales and flip augment for predition. Default: False.
        scales (list|float, optional): Scales for augment. It is valid when `aug_pred` is True. Default: 1.0.
        flip_horizontal (bool, optional): Whether to use flip horizontally augment. It is valid when `aug_pred` is True. Default: True.
        flip_vertical (bool, optional): Whether to use flip vertically augment. It is valid when `aug_pred` is True. Default: False.
        is_slide (bool, optional): Whether to predict by sliding window. Default: False.
        stride (tuple|list, optional): The stride of sliding window, the first is width and the second is height.
            It should be provided when `is_slide` is True.
        crop_size (tuple|list, optional):  The crop size of sliding window, the first is width and the second is height.
            It should be provided when `is_slide` is True.

    """
    utils.utils.load_entire_model(model, model_path)
    model.eval()


    dataset = My_dataset(file_path, mode_forinfer, transform=transforms.transforms)
    loader = DataLoader(
        dataset,
        batch_size=64,
        num_workers=2,
    )    
    tbar = tqdm(loader)

    logger.info("Start to predict...")
    with paddle.no_grad():
        for i, (im, label, im_path) in enumerate(tbar):
            ori_shape = im.shape[:2]

            if aug_pred:
                pred = infer.aug_inference(
                    model,
                    im,
                    ori_shape=ori_shape,
                    transforms=transforms.transforms,
                    scales=scales,
                    flip_horizontal=flip_horizontal,
                    flip_vertical=flip_vertical,
                    is_slide=is_slide,
                    stride=stride,
                    crop_size=crop_size)
            else:
                pred = infer.inference(
                    model,
                    im,
                    ori_shape=ori_shape,
                    transforms=transforms.transforms,
                    is_slide=is_slide,
                    stride=stride,
                    crop_size=crop_size)
            # pred = paddle.squeeze(pred)
            pred = pred.cpu().numpy().astype('uint8')
            label = label.cpu().numpy().astype('uint8')
            
            # multiprocessing.set_start_method('spawn')
            cores = multiprocessing.cpu_count()
            pool = multiprocessing.Pool(cores)

            tmp = zip(im_path,pred,label)
            pool.map(write_img, tmp)
        pool.close()


def write_img(tmp):
    im_path = tmp[0]
    im_file = os.path.basename(im_path)
    save_dir = 'output/result'
    added_saved_dir = os.path.join(save_dir, 'added_prediction')
    org_saved_dir = os.path.join(save_dir, 'result')

    pred = tmp[1].squeeze()
    # label = tmp[2][:,:,::-1]

    # save added image
    # added_image = utils.visualize.visualize(im_path, pred, weight=0.6)
    # added_image_path = os.path.join(added_saved_dir, im_file)
    # mkdir(added_image_path)
    # cv2.imwrite(added_image_path, added_image)

    # img = cv2.imread(im_path, -1)
    # pred = gray2rgb(pred)[:,:,::-1]
    # pred = np.concatenate((img, label, pred), axis=1)
    org_saved_path = os.path.join(
        org_saved_dir, 
        os.path.splitext(im_file)[0] + ".png"
    )
    mkdir(org_saved_path)
    cv2.imwrite(org_saved_path, pred)