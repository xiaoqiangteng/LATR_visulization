#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2022/1/29
# @Author  : Xiaoqiang Teng

import math
import numpy as np
import os
import json
import scipy.interpolate
from collections import OrderedDict


def get_dir_and_file_list(file_path):
    dir_list, file_name_list = [], []
    for _, dirs, files in os.walk(file_path):
        for dir in dirs:
            dir_list.append(dir)

        break

    for root, dirs, files in os.walk(file_path):
        for file_name in files:
            file_name_list.append(file_name)

        break

    return dir_list, file_name_list


def read_file_json(file_name):
    with open(file_name, 'r') as f:
        data_dict = json.load(f)

    return data_dict


def read_file(file_name):
    data_list = []
    with open(file_name, "r") as f:
        for line in f.readlines():
            tags = line.strip().split()

            if len(tags) == 1:
                data_list.append(tags[0])
                continue

            tag_list = []
            for tag in tags:
                tag_list.append(tag)

            data_list.append(tag_list)

    return data_list


def write_file(file_name, data_list):
    with open(file_name, 'w') as f:
        for data in data_list:
            f.write(str(data) + "\n")


def write_file_json(data_dict, output_file_name):
    with open(output_file_name, "w") as f:
        json.dump(data_dict, f, indent=4)


def search_timestamp(ts_list, target_ts, precision=1e4):
    left = 0
    right = len(ts_list)
    while left < right:
        mid = int(left + (right - left) / 2)
        if abs(ts_list[mid] - target_ts) <= precision:
            return mid
        elif ts_list[mid] < target_ts:
            left = mid + 1
        else:
            right = mid
    return -1


def get_box(x, y, z, yaw, length, width, height):
    bbox_center = np.transpose(np.array([x, y, z]))
    transform_matrix = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])

    p_1 = np.transpose(np.array([x - 0.5 * length, y + 0.5 * width, z - 0.5 * height]))
    p_2 = np.transpose(np.array([x - 0.5 * length, y - 0.5 * width, z - 0.5 * height]))
    p_3 = np.transpose(np.array([x + 0.5 * length, y - 0.5 * width, z - 0.5 * height]))
    p_4 = np.transpose(np.array([x + 0.5 * length, y + 0.5 * width, z - 0.5 * height]))
    p_5 = np.transpose(np.array([x - 0.5 * length, y + 0.5 * width, z + 0.5 * height]))
    p_6 = np.transpose(np.array([x - 0.5 * length, y - 0.5 * width, z + 0.5 * height]))
    p_7 = np.transpose(np.array([x + 0.5 * length, y - 0.5 * width, z + 0.5 * height]))
    p_8 = np.transpose(np.array([x + 0.5 * length, y + 0.5 * width, z + 0.5 * height]))
    bbox = [p_1, p_2, p_3, p_4, p_5, p_6, p_7, p_8]

    for i in range(len(bbox)):
        bbox[i] = bbox_center + transform_matrix.dot(bbox[i] - bbox_center)

    bbox_2D = [bbox[0][:2], bbox[1][:2], bbox[2][:2], bbox[3][:2]]

    return bbox, bbox_2D


def get_distance(x_1, y_1, x_2, y_2):
    distance = math.sqrt(math.pow(x_2 - x_1, 2) + math.pow(y_2 - y_1, 2))
    return distance


def get_data_interpolation(input, input_timestamp, output_timestamp):
    """
    This function interpolate n-d vectors (despite the '3d' in the function name) into the output time stamps.
    
    Args:
        input: Nxd array containing N d-dimensional vectors.
        input_timestamp: N-sized array containing time stamps for each of the input quaternion.
        output_timestamp: M-sized array containing output time stamps.
    Return:
        quat_inter: Mxd array containing M vectors.
    """
    assert input.shape[0] == input_timestamp.shape[0]
    func = scipy.interpolate.interp1d(input_timestamp, input, axis=0, fill_value="extrapolate")
    interpolated = func(output_timestamp)
    return interpolated
