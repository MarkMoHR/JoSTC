import os
import random

os.environ['CUDA_VISIBLE_DEVICES'] = '7'

import pydiffvg
import torch

import numpy as np
import math
import shutil
from PIL import Image
import matplotlib.pyplot as plt

import xml.etree.ElementTree as ET
from xml.dom import minidom
from svg.path import parse_path, path

from dataset_utils.dataset_util import draw_segment_jointly, draw_segment_separately


def find_top_k(num_list, top_k):
    sorted_list = np.argsort(num_list)
    tops = np.zeros_like(sorted_list)
    for i in range(len(sorted_list)):
        tops[sorted_list[i]] = i
    tops += 1
    return tops > len(tops) - top_k


def affine_trans(points, trans_matrix):
    new_points = []

    for point in points:
        point_ = np.stack([point[0], point[1], 1.0])
        new_point = np.matmul(trans_matrix, point_)[0:2]
        new_points.append(new_point)

    return new_points


def translation(points, offset):
    trans_matrix = np.array([[1, 0, offset[0]],
                             [0, 1, offset[1]],
                             [0, 0, 1]], dtype=np.float32)
    points_trans = affine_trans(points, trans_matrix)
    return points_trans


def rotation(points, angle_, center):
    angle = angle_ / 180.0 * math.pi

    translate_matrix = np.array([[1, 0, -center[0]],
                                 [0, 1, -center[1]],
                                 [0, 0, 1]], dtype=np.float32)

    rot_matrix = np.array([[math.cos(angle), math.sin(angle), 0],
                           [-math.sin(angle), math.cos(angle), 0],
                           [0, 0, 1]], dtype=np.float32)

    translate_reverse_matrix = np.array([[1, 0, center[0]],
                                         [0, 1, center[1]],
                                         [0, 0, 1]], dtype=np.float32)

    combined_matrix = np.matmul(translate_reverse_matrix, np.matmul(rot_matrix, translate_matrix))
    points_rotation = affine_trans(points, combined_matrix)

    return points_rotation


def rotation_global(strokes_data, angle_thresholds):
    trans_strokes_data = []

    angle = random.random() * (angle_thresholds[1] - angle_thresholds[0]) + angle_thresholds[0]
    all_points = np.concatenate(strokes_data, axis=0)  # (N', 2)
    center_index = random.randint(0, all_points.shape[0] - 1)
    center = all_points[center_index]

    for points in strokes_data:
        trans_points = rotation(points, angle, center=center)
        trans_strokes_data.append(np.stack(trans_points, axis=0).astype(np.float32))

    return trans_strokes_data


def scaling(points, scales, center):
    translate_matrix = np.array([[1, 0, -center[0]],
                                 [0, 1, -center[1]],
                                 [0, 0, 1]], dtype=np.float32)

    scale_matrix = np.array([[scales[0], 0, 0],
                             [0, scales[1], 0],
                             [0, 0, 1]], dtype=np.float32)

    translate_reverse_matrix = np.array([[1, 0, center[0]],
                                         [0, 1, center[1]],
                                         [0, 0, 1]], dtype=np.float32)

    combined_matrix = np.matmul(translate_reverse_matrix, np.matmul(scale_matrix, translate_matrix))
    points_scaling = affine_trans(points, combined_matrix)

    return points_scaling


def scaling_global(strokes_data, scale_thresholds):
    trans_strokes_data = []

    scale_x = random.random() * (scale_thresholds[1] - scale_thresholds[0]) + scale_thresholds[0]
    scale_y = random.random() * (scale_thresholds[1] - scale_thresholds[0]) + scale_thresholds[0]

    all_points = np.concatenate(strokes_data, axis=0)  # (N', 2)
    center_index = random.randint(0, all_points.shape[0] - 1)
    center = all_points[center_index]

    for points in strokes_data:
        trans_points = scaling(points, (scale_x, scale_y), center=center)
        trans_strokes_data.append(np.stack(trans_points, axis=0).astype(np.float32))

    return trans_strokes_data


def shearing(points, angles, center):
    angle_x = angles[0] / 180.0 * math.pi
    angle_y = angles[1] / 180.0 * math.pi

    translate_matrix = np.array([[1, 0, -center[0]],
                                 [0, 1, -center[1]],
                                 [0, 0, 1]], dtype=np.float32)

    shear_matrix = np.array([[1, math.tan(angle_x), 0],
                             [math.tan(angle_y), 1, 0],
                             [0, 0, 1]], dtype=np.float32)

    translate_reverse_matrix = np.array([[1, 0, center[0]],
                                         [0, 1, center[1]],
                                         [0, 0, 1]], dtype=np.float32)

    combined_matrix = np.matmul(translate_reverse_matrix, np.matmul(shear_matrix, translate_matrix))
    points_shear = affine_trans(points, combined_matrix)

    return points_shear


def shearing_global(strokes_data, angle_thresholds):
    trans_strokes_data = []

    angle = random.random() * (angle_thresholds[1] - angle_thresholds[0]) + angle_thresholds[0]
    angles = (angle, 0.0) if random.randint(0, 1) else (0.0, angle)

    all_points = np.concatenate(strokes_data, axis=0)  # (N', 2)
    center_index = random.randint(0, all_points.shape[0] - 1)
    center = all_points[center_index]

    for points in strokes_data:
        trans_points = shearing(points, angles, center=center)
        trans_strokes_data.append(np.stack(trans_points, axis=0).astype(np.float32))

    return trans_strokes_data


def should_do(prob):
    return random.random() <= prob


def choose_which_index(prob_list):
    prob_list_cumsum = np.cumsum(prob_list) / np.sum(prob_list)
    select_prob = random.random()
    for i in range(len(prob_list_cumsum)):
        if select_prob <= prob_list_cumsum[i]:
            return i


def sketch_global_deformation(points_list):
    ## Hyper-parameters
    global_deform_prob = 0.95
    global_n_way_deform_probs = [0.4, 0.35, 0.25]
    global_deform_types = ['rotation', 'scaling', 'shearing']

    global_rotation_angle_thresholds = [-30.0, 30.0]
    global_scaling_thresholds = [0.7, 2.0]
    global_shearing_thresholds = [-20.0, 20.0]

    if should_do(global_deform_prob):
        n_way_deform = choose_which_index(global_n_way_deform_probs) + 1  # {1, 2, 3}
        random.shuffle(global_deform_types)
        selected_deform_types = global_deform_types[:n_way_deform]

        trans_points_list = [item for item in points_list]

        for selected_deform_type in selected_deform_types:
            if selected_deform_type == 'rotation':
                trans_points_list = rotation_global(trans_points_list, global_rotation_angle_thresholds)

            if selected_deform_type == 'scaling':
                trans_points_list = scaling_global(trans_points_list, global_scaling_thresholds)

            if selected_deform_type == 'shearing':
                trans_points_list = shearing_global(trans_points_list, global_shearing_thresholds)

        return trans_points_list
    else:
        return points_list


def dfs(adj_mat, curr_index, done_mat):
    done_mat[curr_index] = 1
    adj_indices = np.argwhere(adj_mat[curr_index] == 1)
    adj_indices_return = [curr_index]
    for adj_index in adj_indices:
        if not done_mat[adj_index[0]]:
            result = dfs(adj_mat, adj_index[0], done_mat)
            adj_indices_return += result[0]
            done_mat = result[1]
    return adj_indices_return, done_mat


def indices_clustering2(indices):
    indices_np = np.stack(indices)
    indices_np_rev = np.concatenate([indices_np[:, 1:2], indices_np[:, 0:1]], axis=-1)
    indices_np = np.concatenate([indices_np, indices_np_rev], axis=0)

    max_index = np.max(indices_np)
    adj_mat = np.zeros(shape=(max_index + 1, max_index + 1), dtype=np.float32)
    done_mat = np.zeros(shape=(max_index + 1), dtype=np.float32)
    for pos in indices_np:
        adj_mat[tuple(pos)] = 1

    cluster_list = []
    for i in range(max_index + 1):
        if not done_mat[i]:
            indices_return, done_mat = dfs(adj_mat, i, done_mat)
            indices_return.sort()
            cluster_list.append(indices_return)

    return cluster_list


def connect_close_strokes(raw_points_list, connect_threshold):
    endpoints_list = np.stack([np.concatenate([item[0], item[-1]]) for item in raw_points_list], axis=0)  # (N, 4)
    start_points_list = endpoints_list[:, 0:2]  # (N, 2)
    end_points_list = endpoints_list[:, 2:4]  # (N, 2)

    start_start_dist = np.expand_dims(start_points_list, axis=1) - np.expand_dims(start_points_list, axis=0)  # (N, N, 2)
    start_start_dist = np.sqrt(np.sum(np.power(start_start_dist, 2), axis=-1))  # (N, N)

    end_end_dist = np.expand_dims(end_points_list, axis=1) - np.expand_dims(end_points_list, axis=0)  # (N, N, 2)
    end_end_dist = np.sqrt(np.sum(np.power(end_end_dist, 2), axis=-1))  # (N, N)

    start_end_dist = np.expand_dims(start_points_list, axis=1) - np.expand_dims(end_points_list, axis=0)  # (N, N, 2)
    start_end_dist = np.sqrt(np.sum(np.power(start_end_dist, 2), axis=-1))  # (N, N)
    start_end_dist = start_end_dist * (1.0 - np.eye(start_end_dist.shape[0]))  # (N, N)

    dist_mat = np.stack([start_start_dist, end_end_dist, start_end_dist], axis=-1)  # (N, N, 3)
    dist_mat = np.min(dist_mat, axis=-1)  # (N, N)
    # dist_mat = dist_mat + np.eye(dist_mat.shape[0]) * (connect_threshold + 1)

    connect_indices = np.argwhere(dist_mat <= connect_threshold)  # list of (2)
    cluster_indices = indices_clustering2(connect_indices)  # list of indices_list
    return cluster_indices


def sketch_local_deformation(points_list):
    ## Hyper-parameters
    connect_distance_threshold = 10

    top_k_stroke_for_special = 2

    local_deform_prob_map = {'normal': 0.8, 'special': 0.6}
    local_n_way_deform_probs = [0.4, 0.3, 0.2, 0.1]
    local_deform_types = ['translation', 'rotation', 'scaling', 'shearing']

    local_translation_thresholds_map = {'normal': [-10.0, 10.0], 'special': [-5.0, 5.0]}
    local_rotation_angle_thresholds_map = {'normal': [-15.0, 15.0], 'special': [-5.0, 5.0]}
    local_scaling_thresholds_map = {'normal': [0.8, 1.3], 'special': [0.9, 1.2]}
    local_shearing_thresholds_map = {'normal': [-10.0, 10.0], 'special': [-5.0, 5.0]}

    trans_points_list = [[] for _ in range(len(points_list))]

    # generate stroke clusters first
    cluster_indices = connect_close_strokes(points_list, connect_distance_threshold)  # [[0, 1, 3], [2], [4, 5], ...]

    cluster_strokes_len = []
    for cluster_i in range(len(cluster_indices)):
        cluster_stroke_ids = cluster_indices[cluster_i]
        cluster_stroke_len = np.sum([len(points_list[s_i]) for s_i in cluster_stroke_ids]) - (len(cluster_stroke_ids) - 1)
        cluster_strokes_len.append(cluster_stroke_len)
    is_top_k = find_top_k(cluster_strokes_len, top_k_stroke_for_special)

    # deform each cluster
    for cluster_i in range(len(cluster_indices)):
        cluster_stroke_ids = cluster_indices[cluster_i]
        cluster_points = np.concatenate([points_list[s_i] for s_i in cluster_stroke_ids], axis=0)

        stroke_type = 'special' if is_top_k[cluster_i] else 'normal'

        local_deform_prob = local_deform_prob_map[stroke_type]
        local_translation_thresholds = local_translation_thresholds_map[stroke_type]
        local_rotation_angle_thresholds = local_rotation_angle_thresholds_map[stroke_type]
        local_scaling_thresholds = local_scaling_thresholds_map[stroke_type]
        local_shearing_thresholds = local_shearing_thresholds_map[stroke_type]

        if should_do(local_deform_prob):
            n_way_deform = choose_which_index(local_n_way_deform_probs) + 1  # {1, 2, 3}
            random.shuffle(local_deform_types)
            selected_deform_types = local_deform_types[:n_way_deform]

            trans_x = random.random() * (local_translation_thresholds[1] - local_translation_thresholds[0]) + local_translation_thresholds[0]
            trans_y = random.random() * (local_translation_thresholds[1] - local_translation_thresholds[0]) + local_translation_thresholds[0]

            rot_angle = random.random() * (local_rotation_angle_thresholds[1] - local_rotation_angle_thresholds[0]) + local_rotation_angle_thresholds[0]
            rot_center_index = random.randint(0, len(cluster_points) - 1)
            rot_center = cluster_points[rot_center_index]

            scale_x = random.random() * (local_scaling_thresholds[1] - local_scaling_thresholds[0]) + local_scaling_thresholds[0]
            scale_y = random.random() * (local_scaling_thresholds[1] - local_scaling_thresholds[0]) + local_scaling_thresholds[0]
            scale_center_index = random.randint(0, len(cluster_points) - 1)
            scale_center = cluster_points[scale_center_index]

            sheer_angle = random.random() * (local_shearing_thresholds[1] - local_shearing_thresholds[0]) + local_shearing_thresholds[0]
            sheer_angles = (sheer_angle, 0.0) if random.randint(0, 1) else (0.0, sheer_angle)
            sheer_center_index = random.randint(0, len(cluster_points) - 1)
            sheer_center = cluster_points[sheer_center_index]

            for stroke_i in cluster_stroke_ids:
                trans_points = [item for item in points_list[stroke_i]]

                for selected_deform_type in selected_deform_types:
                    if selected_deform_type == 'translation':
                        trans_points = translation(trans_points, (trans_x, trans_y))

                    if selected_deform_type == 'rotation':
                        trans_points = rotation(trans_points, rot_angle, rot_center)

                    if selected_deform_type == 'scaling':
                        trans_points = scaling(trans_points, (scale_x, scale_y), scale_center)

                    if selected_deform_type == 'shearing':
                        trans_points = shearing(trans_points, sheer_angles, sheer_center)

                trans_points_list[stroke_i] = np.stack(trans_points, axis=0).astype(np.float32)
        else:
            for stroke_i in cluster_stroke_ids:
                trans_points_list[stroke_i] = points_list[stroke_i]

    for item in trans_points_list:
        assert len(item) != 0
    return trans_points_list


def check_out_of_bound(points_list, image_size):
    all_points = np.concatenate(points_list, axis=0)
    return (all_points > image_size).any() or (all_points < 0).any()


def generate_paired_deformation(data_base):
    '''
    For generating the paired deformation of the reference sketches.
    '''
    canvas_size = 320
    max_seq_num = 40
    stroke_thickness = 1.2

    # validation
    pydiffvg.set_use_gpu(torch.cuda.is_available())

    background_ = torch.ones(canvas_size, canvas_size, 4)
    render_ = pydiffvg.RenderFunction.apply

    # split_nums = {'train': 10000, 'valid': 1000}
    splits = ['train', 'valid']

    for split in splits:
        split_txt = os.path.join(data_base, split + '.txt')
        save_base_black = os.path.join(data_base, split, 'black')
        save_base_color_joint = os.path.join(data_base, split, 'color_joint')
        save_base_color_separate = os.path.join(data_base, split, 'color_separate')
        save_base_parameter = os.path.join(data_base, split, 'parameter')

        with open(split_txt, "r") as f:
            all_lines = f.readlines()
            for l_i, line in enumerate(all_lines):
                img_name = 'example-' + str(l_i) + '.png'
                print('Processing', l_i, '/', len(all_lines))
                sketch_npz_path = os.path.join(save_base_parameter, 'example-' + str(l_i) + '-original.npz')
                npz = np.load(sketch_npz_path, encoding='latin1', allow_pickle=True)
                strokes_data = npz['strokes_data']  # list of point_list in (N, 2), in canvas size

                while True:
                    trans_strokes_data = sketch_global_deformation(strokes_data)
                    if check_out_of_bound(trans_strokes_data, canvas_size):
                        continue

                    trans_strokes_data = sketch_local_deformation(trans_strokes_data)
                    if check_out_of_bound(trans_strokes_data, canvas_size):
                        continue

                    # saving
                    save_npz_path = os.path.join(save_base_parameter, 'example-' + str(l_i) + '-transformed.npz')
                    np.savez(save_npz_path, strokes_data=trans_strokes_data, canvas_size=canvas_size)

                    black_stroke_img = draw_segment_jointly(trans_strokes_data, canvas_size, stroke_thickness, render_, background_, is_black=True)
                    black_stroke_img_save_path = os.path.join(save_base_black, 'example-' + str(l_i) + '-transformed.png')
                    pydiffvg.imwrite(black_stroke_img.cpu(), black_stroke_img_save_path, gamma=1.0)

                    color_stroke_img = draw_segment_jointly(trans_strokes_data, canvas_size, stroke_thickness, render_, background_, is_black=False)
                    color_stroke_img_save_path = os.path.join(save_base_color_joint, 'example-' + str(l_i) + '-transformed.png')
                    pydiffvg.imwrite(color_stroke_img.cpu(), color_stroke_img_save_path, gamma=1.0)

                    color_stroke_img = draw_segment_separately(trans_strokes_data, canvas_size, stroke_thickness,
                                                               max_seq_num, render_, background_, is_black=False)
                    color_stroke_img_save_path = os.path.join(save_base_color_separate, 'example-' + str(l_i) + '-transformed.png')
                    pydiffvg.imwrite(color_stroke_img.cpu(), color_stroke_img_save_path, gamma=1.0)

                    break


def load_data(data_base):
    canvas_size = 320
    max_seq_num = 40
    stroke_thickness = 1.2

    # validation
    pydiffvg.set_use_gpu(torch.cuda.is_available())

    background_ = torch.ones(canvas_size, canvas_size, 4)
    render_ = pydiffvg.RenderFunction.apply

    split_nums = {'train': 10000, 'valid': 1000}
    splits = ['valid']

    for split in splits:
        split_num = split_nums[split]

        save_base_color_separate = os.path.join(data_base, split, 'vector_vis')
        parameter_base = os.path.join(data_base, split, 'parameter')

        for l_i in range(split_num):
            print('Processing', l_i, '/', split_num)
            sketch_npz_path = os.path.join(parameter_base, 'example-' + str(l_i) + '-original.npz')
            npz = np.load(sketch_npz_path, encoding='latin1', allow_pickle=True)
            strokes_data_ref = npz['strokes_data']  # list of point_list in (N, 2), in canvas size

            sketch_npz_path = os.path.join(parameter_base, 'example-' + str(l_i) + '-transformed.npz')
            npz = np.load(sketch_npz_path, encoding='latin1', allow_pickle=True)
            strokes_data_tar = npz['strokes_data']  # list of point_list in (N, 2), in canvas size

            # saving
            color_stroke_img = draw_segment_separately(strokes_data_ref, canvas_size, stroke_thickness, max_seq_num, render_, background_, is_black=False)
            color_stroke_img_save_path = os.path.join(save_base_color_separate, 'example-' + str(l_i) + '-original.png')
            pydiffvg.imwrite(color_stroke_img.cpu(), color_stroke_img_save_path, gamma=1.0)

            color_stroke_img = draw_segment_separately(strokes_data_tar, canvas_size, stroke_thickness, max_seq_num, render_, background_, is_black=False)
            color_stroke_img_save_path = os.path.join(save_base_color_separate, 'example-' + str(l_i) + '-transformed.png')
            pydiffvg.imwrite(color_stroke_img.cpu(), color_stroke_img_save_path, gamma=1.0)


if __name__ == '__main__':
    data_base = '/nfs/sketch-tracing-datasets/TU-Berlin'

    # generate_paired_deformation(data_base)

    load_data(data_base)
