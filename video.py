# coding: utf-8
'''
载入显著图特征，用vgg16提取
'''

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import os
import cv2
import h5py
import numpy as np
import skimage
import torch
from torch.autograd import Variable
from model import AppearanceEncoder, MotionEncoder
from args import video_root, video_sort_lambda
from args import feature_h5_path, feature_h5_feats, feature_h5_lens
from args import max_frames, feature_size


def sample_frames(video_path, train=True):
    '''
    对视频帧进行采样，减少计算量。等间隔地取max_frames帧
    '''
    try:
        cap = cv2.VideoCapture(video_path)
    except:
        print('Can not open %s.' % video_path)
        pass

    frames = []
    frame_count = 0

    while True:
        ret, frame = cap.read()
        # 把BGR的图片转换成RGB的图片，因为之后的模型用的是RGB格式
        if ret is False:
            break
        frame = frame[:, :, ::-1]
        frames.append(frame)
        frame_count += 1

    indices = np.linspace(8, frame_count - 7, max_frames, endpoint=False, dtype=int)

    frames = np.array(frames)
    frame_list = frames[indices]
    clip_list = []
    for index in indices:
        clip_list.append(frames[index - 8: index + 8])
    clip_list = np.array(clip_list)
    return frame_list, clip_list, frame_count


def resize_frame(image, target_height=224, target_width=224):
    if len(image.shape) == 2:
        # 把单通道的灰度图复制三遍变成三通道的图片
        image = np.tile(image[:, :, None], 3)
    elif len(image.shape) == 4:
        image = image[:, :, :, 0]

    height, width, channels = image.shape
    if height == width:
        resized_image = cv2.resize(image, (target_height, target_width))
    elif height < width:
        resized_image = cv2.resize(image, (int(width * target_height / height),
                                           target_width))
        cropping_length = int((resized_image.shape[1] - target_height) / 2)
        resized_image = resized_image[:,
                                      cropping_length:resized_image.shape[1] - cropping_length]
    else:
        resized_image = cv2.resize(image, (target_height,
                                           int(height * target_width / width)))
        cropping_length = int((resized_image.shape[0] - target_width) / 2)
        resized_image = resized_image[cropping_length:
                                      resized_image.shape[0] - cropping_length]
    return cv2.resize(resized_image, (target_height, target_width))


def preprocess_frame(image, target_height=224, target_width=224):
    image = resize_frame(image, target_height, target_width)
    image = skimage.img_as_float(image).astype(np.float32)
    # 根据在ILSVRC数据集上的图像的均值（RGB格式）进行白化
    image -= np.array([0.485, 0.456, 0.406])
    image /= np.array([0.229, 0.224, 0.225])
    return image


def extract_features(aencoder, mencoder):
    # 读取视频列表，让视频按照id升序排列
    videos = sorted(os.listdir(video_root), key=video_sort_lambda)
    nvideos = len(videos)

    # 创建保存视频特征的hdf5文件
    if os.path.exists(feature_h5_path):
        # 如果hdf5文件已经存在，说明之前处理过，或许是没有完全处理完
        # 使用r+ (read and write)模式读取，以免覆盖掉之前保存好的数据
        h5 = h5py.File(feature_h5_path, 'r+')
        dataset_feats = h5[feature_h5_feats]
        dataset_lens = h5[feature_h5_lens]
    else:
        h5 = h5py.File(feature_h5_path, 'w')
        dataset_feats = h5.create_dataset(feature_h5_feats,
                                          (nvideos, max_frames, feature_size),
                                          dtype='float32')
        dataset_lens = h5.create_dataset(feature_h5_lens, (nvideos,), dtype='int')

    for i, video in enumerate(videos):
        print(video, end=' ')
        video_path = os.path.join(video_root, video)
        # 提取视频帧以及视频小块
        frame_list, clip_list, frame_count = sample_frames(video_path, train=True)
        print(frame_count)

        # 把图像做一下处理，然后转换成（batch, channel, height, width）的格式
        frame_list = np.array([preprocess_frame(x) for x in frame_list])
        frame_list = frame_list.transpose((0, 3, 1, 2))
        frame_list = Variable(torch.from_numpy(frame_list), volatile=True).cuda()

        # 视频特征的shape是max_frames x (2048 + 4096)
        # 如果帧的数量小于max_frames，则剩余的部分用0补足
        feats = np.zeros((max_frames, feature_size), dtype='float32')

        # 先提取表观特征
        af = aencoder(frame_list)

        # 再提取动作特征
        clip_list = np.array([[resize_frame(x, 112, 112)
                               for x in clip] for clip in clip_list])
        clip_list = clip_list.transpose(0, 4, 1, 2, 3).astype(np.float32)
        clip_list = Variable(torch.from_numpy(clip_list), volatile=True).cuda()
        mf = mencoder(clip_list)

        # 合并表观和动作特征
        feats[:frame_count, :] = torch.cat([af, mf], dim=1).data.cpu().numpy()
        dataset_feats[i] = feats
        dataset_lens[i] = frame_count


def main():
    aencoder = AppearanceEncoder()
    aencoder.eval()
    aencoder.cuda()

    mencoder = MotionEncoder()
    mencoder.eval()
    mencoder.cuda()

    extract_features(aencoder, mencoder)


if __name__ == '__main__':
    main()
