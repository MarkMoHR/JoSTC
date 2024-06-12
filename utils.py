import os
import random
import numpy as np
import math
from PIL import Image
from scipy.ndimage import distance_transform_edt

import torch
import torch.nn.functional as F
import pydiffvg

from hparam import HParams


def copy_hparams(hparams):
    """Return a copy of an HParams instance."""
    return HParams(**hparams.values())


class LineDataLoader(object):
    def __init__(self,
                 dataset_name,
                 dataset_base,
                 output_base,
                 stroke_thickness):
        self.dataset_name = dataset_name
        self.dataset_base = dataset_base
        self.output_base = output_base
        self.stroke_thickness = stroke_thickness

        if self.dataset_name == 'clean' or self.dataset_name == 'rough':
            self.load_target_point = False
        # elif self.dataset_name == 'TU-Berlin' or self.dataset_name == 'TU-Derlin' or self.dataset_name == 'TU-Refined':
        #     self.ref_tar_split_names = ['original', 'transformed']
        #     self.load_target_point = True
        else:
            raise Exception('Unknown dataset_name', self.dataset_name)

    def load_image(self, img_path):
        image = Image.open(img_path).convert("RGB")
        image = np.array(image, dtype=np.float32)  # (H, W, 3), [0.0-strokes, 255.0-BG]
        image = image[:, :, 0] / 255.0  # (H, W), [0.0-strokes, 1.0-BG]
        return image

    def load_stroke_parameter(self, npz_path):
        npz = np.load(npz_path, encoding='latin1', allow_pickle=True)
        strokes_data = npz['strokes_data']  # list of point_list in (N, 2), in canvas size
        return strokes_data

    def process_stroke_parameter_ref(self, parameters, image_size):
        endpoints = []
        starting_states = []
        window_size = []
        stroke_segment_images = []

        for stroke_i, point_list in enumerate(parameters):
            segment_num = (len(point_list) - 1) // 3

            single_stroke_endpoints = [np.array(point_list[i * 3], dtype=np.float32) / float(image_size) for i in range(segment_num)]
            endpoints += single_stroke_endpoints

            single_stroke_starting_states = [1.0] + [0.0 for _ in range(segment_num - 1)]
            starting_states += single_stroke_starting_states

            single_stroke_window_size = [np.array(point_list[i * 3], dtype=np.float32) for i in range(segment_num + 1)]
            single_stroke_window_size = np.stack(single_stroke_window_size, axis=0)  # (segment_num + 1, 2), full size
            single_stroke_window_size_prev = single_stroke_window_size[0:segment_num, :]  # (segment_num, 2), full size
            single_stroke_window_size_next = single_stroke_window_size[1:segment_num+1, :]  # (segment_num, 2), full size
            single_stroke_window_size_dist = np.abs(single_stroke_window_size_prev - single_stroke_window_size_next)  # (segment_num, 2), full size
            single_stroke_window_size = np.max(single_stroke_window_size_dist, axis=-1) * 2.0 / float(image_size)  # (segment_num), [0.0, 1.0]
            single_stroke_window_size = single_stroke_window_size.tolist()
            window_size += single_stroke_window_size

            single_stroke_segment_images = self.render_stroke_images_tensor(point_list, image_size, self.stroke_thickness)
            # single_stroke_segment_images = self.load_stroke_images(img_idx, stroke_i, segment_num, reference_split_name)
            # (segment_num, H, W), [0.0-stroke, 1.0-BG]
            stroke_segment_images.append(single_stroke_segment_images)

        endpoints = np.stack(endpoints, axis=0)  # (seq_num, 2), [0.0, 1.0]
        starting_states = np.stack(starting_states, axis=0)  # (seq_num), {1.0, 0.0}
        window_size = np.stack(window_size, axis=0)  # (seq_num), [0.0, 1.0]
        stroke_segment_images = np.concatenate(stroke_segment_images, axis=0)  # (seq_num, H, W), [0.0-stroke, 1.0-BG]

        return endpoints, starting_states, window_size, stroke_segment_images

    def process_stroke_parameter_tar(self, parameters, image_size):
        end_ctrl_points = []
        for stroke_i, point_list in enumerate(parameters):
            segment_num = (len(point_list) - 1) // 3
            single_stroke_end_ctrl_points = [np.array(point_list[i * 3: i * 3 + 4, :], dtype=np.float32) / float(image_size)
                                             for i in range(segment_num)]  # list of (4, 2), [0.0, 1.0]
            end_ctrl_points += single_stroke_end_ctrl_points

        end_ctrl_points = np.stack(end_ctrl_points, axis=0)  # (seq_num, 4, 2), [0.0, 1.0]
        return end_ctrl_points

    def load_stroke_images(self, img_idx, stroke_idx, segment_num, reference_split_name):
        segment_images = []

        for segment_i in range(segment_num):
            segment_img_path = os.path.join(self.dataset_base, 'black-segment', str(img_idx), reference_split_name,
                                            str(stroke_idx) + '-' + str(segment_i) + '.png')
            segment_img = self.load_image(segment_img_path)  # (H, W), [0.0-strokes, 1.0-BG]
            segment_images.append(segment_img)

        segment_images = np.stack(segment_images, axis=0)  # (segment_num, H, W), [0.0-stroke, 1.0-BG]
        return segment_images

    def render_stroke_images_tensor(self, point_list, image_size, stroke_thickness):
        # point_list: (N, 2)
        segment_num = (len(point_list) - 1) // 3
        segment_images = []

        for i in range(segment_num):
            single_segment_params = point_list[i * 3: i * 3 + 4, :]  # (4, 2)

            shapes = []
            shape_groups = []

            num_control_points = torch.tensor([2])
            points = torch.tensor(single_segment_params).float()
            path = pydiffvg.Path(num_control_points=num_control_points,
                                 points=points,
                                 is_closed=False,
                                 stroke_width=torch.tensor(stroke_thickness))
            shapes.append(path)

            path_group = pydiffvg.ShapeGroup(shape_ids=torch.tensor([0]),
                                             fill_color=None,
                                             stroke_color=torch.tensor([0.0, 0.0, 0.0, 1.0]))
            shape_groups.append(path_group)

            scene_args = pydiffvg.RenderFunction.serialize_scene(
                image_size, image_size, shapes, shape_groups)

            background = torch.ones(image_size, image_size, 4)

            render = pydiffvg.RenderFunction.apply
            img = render(image_size,  # width
                         image_size,  # height
                         2,  # num_samples_x
                         2,  # num_samples_y
                         0,  # seed
                         background,  # background_image
                         *scene_args)
            segment_img = img[:, :, 0]  # (H, W), [0.0-stroke, 1.0-BG]
            segment_images.append(segment_img)

        # for i in range(self.max_segment_num - segment_num):
        #     padded_img = torch.ones(size=(image_size, image_size), dtype=torch.float32).cuda()  # (H, W), [0.0-stroke, 1.0-BG]
        #     segment_images.append(padded_img)

        segment_images = torch.stack(segment_images, dim=0)  # (segment_num, H, W), [0.0-stroke, 1.0-BG]
        segment_images = segment_images.cpu().data.numpy()
        return segment_images

    def get_batch(self, use_cuda, reference_img_name, target_img_name, is_manual_reference):
        reference_image_batch = []
        target_image_batch = []
        reference_segment_image_batch = []
        reference_dot_patch_batch = []
        reference_endpoints_batch = []
        target_points_batch = []
        starting_states_batch = []
        base_window_size_batch = []

        reference_image_path = os.path.join(self.dataset_base, 'raster', reference_img_name)
        target_image_path = os.path.join(self.dataset_base, 'raster', target_img_name)
        if is_manual_reference:
            reference_stroke_path = os.path.join(self.dataset_base, 'parameter', reference_img_name[:reference_img_name.find('.')] + '.npz')
        else:
            reference_stroke_path = os.path.join(self.output_base, 'parameter', reference_img_name[:reference_img_name.find('.')] + '.npz')

        # if self.dataset_name == 'TU-Berlin' or self.dataset_name == 'TU-Derlin' or self.dataset_name == 'TU-Refined':
        #     reference_image_path = os.path.join(self.dataset_base, 'black', 'example-' + str(selected_index) + '-' + self.ref_tar_split_names[0] + '.png')
        #     reference_stroke_path = os.path.join(self.dataset_base, 'parameter', 'example-' + str(selected_index) + '-' + self.ref_tar_split_names[0] + '.npz')
        #     target_image_path = os.path.join(self.dataset_base, 'black', 'example-' + str(selected_index) + '-' + self.ref_tar_split_names[1] + '.png')
        #     target_stroke_path = os.path.join(self.dataset_base, 'parameter', 'example-' + str(selected_index) + '-' + self.ref_tar_split_names[1] + '.npz')
        # else:
        #     raise Exception('Unknown dataset_name', self.dataset_name)

        dot_image_path = os.path.join('dataset_utils', 'ref-dot.png')

        reference_image = self.load_image(reference_image_path)  # (H, W), [0.0-strokes, 1.0-BG]
        dot_image = self.load_image(dot_image_path)  # (H_c, W_c), [0.0-strokes, 1.0-BG]
        target_image = self.load_image(target_image_path)  # (H, W), [0.0-strokes, 1.0-BG]
        reference_image_batch.append(reference_image)
        reference_dot_patch_batch.append(dot_image)
        target_image_batch.append(target_image)

        image_size = reference_image.shape[0]

        reference_stroke_raw = self.load_stroke_parameter(reference_stroke_path)
        endpoints, starting_states, window_sizes, stroke_segment_images = \
            self.process_stroke_parameter_ref(reference_stroke_raw, image_size)
        # endpoints: (seq_num, 2), [0.0, 1.0]
        # starting_states: (seq_num), {1.0, 0.0}
        # window_sizes: (seq_num), [0.0, 1.0]
        # stroke_segment_images: (seq_num, H, W), [0.0-stroke, 1.0-BG]
        reference_segment_image_batch.append(stroke_segment_images)
        reference_endpoints_batch.append(endpoints)
        starting_states_batch.append(starting_states)
        base_window_size_batch.append(window_sizes)

        if self.load_target_point:
            target_stroke_raw = self.load_stroke_parameter(target_stroke_path)
            all_points_tar = self.process_stroke_parameter_tar(target_stroke_raw, image_size)
            # all_points_tar: (seq_num, 4, 2), [0.0, 1.0]
            target_points_batch.append(all_points_tar)

        reference_image_batch = np.expand_dims(np.stack(reference_image_batch, axis=0), axis=-1)  # (N, H, W, 1), [0.0-strokes, 1.0-BG]
        reference_dot_patch_batch = np.stack(reference_dot_patch_batch, axis=0)  # (N, H_c, W_c), [0.0-strokes, 1.0-BG]
        target_image_batch = np.expand_dims(np.stack(target_image_batch, axis=0), axis=-1)  # (N, H, W, 1), [0.0-strokes, 1.0-BG]
        reference_segment_image_batch = np.stack(reference_segment_image_batch, axis=0)  # (N, seq_num, H, W), [0.0-stroke, 1.0-BG]
        reference_endpoints_batch = np.stack(reference_endpoints_batch, axis=0)  # (N, seq_num, 2), [0.0, 1.0]
        starting_states_batch = np.stack(starting_states_batch, axis=0)  # (N, seq_num), {1.0, 0.0}
        base_window_size_batch = np.stack(base_window_size_batch, axis=0)  # (N, seq_num), [0.0, 1.0]

        ## convert to tensor
        reference_image_batch = torch.tensor(reference_image_batch).float()
        reference_dot_patch_batch = torch.tensor(reference_dot_patch_batch).float()
        target_image_batch = torch.tensor(target_image_batch).float()
        reference_segment_image_batch = torch.tensor(reference_segment_image_batch).float()
        reference_endpoints_batch = torch.tensor(reference_endpoints_batch).float()
        starting_states_batch = torch.tensor(starting_states_batch).float()
        base_window_size_batch = torch.tensor(base_window_size_batch).float()

        if use_cuda:
            reference_image_batch = reference_image_batch.cuda()
            reference_dot_patch_batch = reference_dot_patch_batch.cuda()
            target_image_batch = target_image_batch.cuda()
            reference_segment_image_batch = reference_segment_image_batch.cuda()
            reference_endpoints_batch = reference_endpoints_batch.cuda()
            starting_states_batch = starting_states_batch.cuda()
            base_window_size_batch = base_window_size_batch.cuda()

        if self.load_target_point:
            target_points_batch = np.stack(target_points_batch, axis=0)  # (N, seq_num, 4, 2), [0.0, 1.0]
            target_points_batch = torch.tensor(target_points_batch).float()
            if use_cuda:
                target_points_batch = target_points_batch.cuda()
        else:
            target_points_batch = None

        return reference_image_batch, target_image_batch, reference_segment_image_batch, reference_dot_patch_batch, \
            reference_endpoints_batch, starting_states_batch, base_window_size_batch, target_points_batch


def load_dataset(model_params):
    data_base = os.path.join(model_params.dataset_base, model_params.data_type)
    output_base = os.path.join(model_params.inference_root, model_params.data_type)
    valid_model_params = copy_hparams(model_params)

    assert model_params.window_size_scaling_ref == 1.5 and model_params.window_size_scaling_init_tar == 1.5
    val_set = LineDataLoader(model_params.data_type, dataset_base=data_base, output_base=output_base,
                             stroke_thickness=valid_model_params.stroke_thickness)
    return val_set


def normalize_image_m1to1(in_img_0to1):
    norm_img_m1to1 = torch.mul(in_img_0to1, 2.0)
    norm_img_m1to1 = torch.sub(norm_img_m1to1, 1.0)
    return norm_img_m1to1


def image_cropping_stn(cursor_position, input_img, image_size, window_sizes_in, raster_size, rotation_angle=None):
    """
    :param cursor_position: (N, 1, 2), float type, in size [0.0, 1.0)
    :param input_img: [0.0-stroke, 1.0-BG]
    :param window_sizes_in: (N, 1, 2), float type, in full size
    :param rotation_angle: (N, 1), in degree
    """
    center_pos = cursor_position.squeeze(dim=1)  # (N, 2), float type, in size [0.0, 1.0)
    window_size = window_sizes_in.squeeze(dim=1)  # (N, 2), float type, in full size
    img = input_img.permute(0, 3, 1, 2)

    center_pos_norm = center_pos * 2.0 - 1.0  # (N, 2), [-1.0, 1.0]
    center_pos_x, center_pos_y = torch.split(center_pos_norm, 1, dim=-1)  # (N, 1), [-1.0, 1.0]
    window_size_norm = window_size / float(image_size)  # (N, 2), [0.0, 1.0]
    window_size_x, window_size_y = torch.split(window_size_norm, 1, dim=-1)  # (N, 1), [0.0, 1.0]

    batch_size = img.size(0)
    channel = img.size(1)

    ones_tensor = torch.ones_like(center_pos_x)  # (N, 1)
    zeros_tensor = torch.zeros_like(center_pos_x)  # (N, 1)

    # shifting
    translate_matrix = torch.cat([ones_tensor, zeros_tensor, center_pos_x,
                                  zeros_tensor, ones_tensor, center_pos_y,
                                  zeros_tensor, zeros_tensor, ones_tensor], dim=-1)  # (N, 9)
    translate_matrix = torch.reshape(translate_matrix, (-1, 3, 3))  # (N, 3, 3)

    # scaling
    scaling_matrix = torch.cat([window_size_x, zeros_tensor, zeros_tensor,
                                zeros_tensor, window_size_y, zeros_tensor,
                                zeros_tensor, zeros_tensor, ones_tensor], dim=-1)  # (N, 9)
    scaling_matrix = torch.reshape(scaling_matrix, (-1, 3, 3))  # (N, 3, 3)

    # rotation
    if rotation_angle is not None:
        rotation_angle_norm = rotation_angle / 180.0 * math.pi
        rotate_matrix = torch.cat([torch.cos(rotation_angle_norm), torch.sin(rotation_angle_norm), zeros_tensor,
                                   -torch.sin(rotation_angle_norm), torch.cos(rotation_angle_norm), zeros_tensor,
                                   zeros_tensor, zeros_tensor, ones_tensor], dim=-1)  # (N, 9)
        rotate_matrix = torch.reshape(rotate_matrix, (-1, 3, 3))  # (N, 3, 3)
        matrix = torch.matmul(torch.matmul(translate_matrix, scaling_matrix), rotate_matrix)
    else:
        matrix = torch.matmul(translate_matrix, scaling_matrix)
    matrix = matrix[:, 0:2, :]

    affine_grid_points = F.affine_grid(matrix, size=[batch_size, channel, raster_size, raster_size],
                                       align_corners=False)
    rois = F.grid_sample(1.0 - img, affine_grid_points, align_corners=False)
    rois = 1.0 - rois  # (N, C, raster_size, raster_size), [0.0-stroke, 1.0-BG]
    rois = rois.permute(0, 2, 3, 1)  # (N, raster_size, raster_size, C), [0.0-stroke, 1.0-BG]
    return rois


def spatial_transform_reverse_point(points_pos, rotation_angle):
    """
    :param points_pos: (N, 2), [-1.0, 1.0]
    :param rotation_angle: (N, 1), in degree
    """
    ones_tensor = torch.ones_like(rotation_angle)  # (N, 1)
    zeros_tensor = torch.zeros_like(rotation_angle)  # (N, 1)

    rotation_angle_norm_re = -rotation_angle / 180.0 * math.pi
    rotate_matrix = torch.cat([torch.cos(rotation_angle_norm_re), torch.sin(rotation_angle_norm_re), zeros_tensor,
                               -torch.sin(rotation_angle_norm_re), torch.cos(rotation_angle_norm_re), zeros_tensor,
                               zeros_tensor, zeros_tensor, ones_tensor], dim=-1)  # (N, 9)
    rotate_matrix = torch.reshape(rotate_matrix, (-1, 3, 3))  # (N, 3, 3)

    points_pos_full = torch.cat([points_pos, ones_tensor], dim=-1).unsqueeze(dim=1)  # (N, 1, 3)
    points_pos_re = torch.matmul(points_pos_full, rotate_matrix).squeeze(dim=1)  # (N, 1, 3) => (N, 3)
    points_pos_re = points_pos_re[:, 0:2]  # (N, 2), might larger than 1.0
    return points_pos_re


def correspondence_clinging(target_images, pred_positions, img_size, binary_threshold=200.0):
    # target_images: (N, H_c, W_c, 1), [0-stroke, 1-BG]
    # pred_positions: (N, 2), [-1.0, 1.0]
    pred_positions_post = []
    for i in range(pred_positions.shape[0]):
        target_image = target_images[i, :, :, 0]  # (H, W), [0-stroke, 1-BG]
        target_image = np.copy(target_image)
        target_image[target_image < (binary_threshold / 255.0)] = 0.0
        _, dt_inds = distance_transform_edt(target_image, return_indices=True)  # dt_inds: (2, H, W), (y, x)

        pred_pos = pred_positions[i]  # (2), [-1.0, 1.0]
        pred_pos_global = (pred_pos + 1.0) / 2.0 * img_size  # (x, y), in full size
        pred_pos_clinging = dt_inds[:, int(pred_pos_global[1]), int(pred_pos_global[0])]
        pred_pos_clinging = pred_pos_clinging[::-1]  # (x, y), in full size
        pred_pos_clinging = pred_pos_clinging / float(img_size) * 2.0 - 1.0  # (x, y), [-1, 1]
        pred_positions_post.append(pred_pos_clinging)
    pred_positions_post = np.stack(pred_positions_post, axis=0)  # (N, 2), [-1, 1]
    return pred_positions_post


def get_correspondence_window_size(image_size, init_times):
    if init_times == 0.0:
        image_size_gap = [0, 400, 600, 800, 1000]
        correspondence_window_size_gap = [256, 320, 384, 448, 512]
        for i in range(len(image_size_gap)):
            if image_size <= image_size_gap[i]:
                return correspondence_window_size_gap[i-1]
        return correspondence_window_size_gap[-1]
    else:
        return int(image_size * init_times)


def draw_dot(img, dot_pos_norm, color, radius=4):
    img_dot = np.copy(img)
    img_size = img.shape[0]
    dot_pos = (dot_pos_norm + 1.0) / 2.0 * img_size
    dot_left = int(max(dot_pos[0] - radius, 0))
    dot_right = int(min(dot_pos[0] + radius, img_size - 1))
    dot_up = int(max(dot_pos[1] - radius, 0))
    dot_down = int(min(dot_pos[1] + radius, img_size - 1))
    img_dot[dot_up:dot_down, dot_left:dot_right] = color
    return img_dot


def seq_params_to_list(seq_params, starting_states):
    # seq_params: (seq_num, 4, 2), in full size
    # starting_states: (seq_num), {1.0, 0.0}
    def flatten_stroke4(stroke4_list):
        single_stroke_tmp = np.stack(stroke4_list, axis=0)  # (N_segment, 4, 2)
        single_stroke_head = single_stroke_tmp[0, 0:1, :]  # (1, 2)
        single_stroke_other = np.reshape(single_stroke_tmp[:, 1:, :], (-1, 2))  # (N_segment, 3, 2) => (N_segment*3, 2)
        single_stroke_flatten = np.concatenate([single_stroke_head, single_stroke_other], axis=0)  # (N_point, 2)
        return single_stroke_flatten

    params_stroke_list = []  # list of (N_point, 2), in image size
    single_stroke_list = []

    for seq_i in range(len(seq_params)):
        if starting_states[seq_i] == 1 and len(single_stroke_list) > 0:
            single_stroke = flatten_stroke4(single_stroke_list)
            params_stroke_list.append(single_stroke)
            single_stroke_list = []

        single_stroke_list.append(seq_params[seq_i])

        if seq_i == len(seq_params) - 1:
            single_stroke = flatten_stroke4(single_stroke_list)
            params_stroke_list.append(single_stroke)

    return params_stroke_list
