import os
os.environ['CUDA_VISIBLE_DEVICES'] = '7'

import numpy as np
from PIL import Image
import argparse
import sys

sys.path.append("..")

import pydiffvg
import torch

import xml.etree.ElementTree as ET
from xml.dom import minidom
from svg.path import parse_path, path

from dataset_utils.dataset_util import draw_segment_jointly, draw_segment_separately


invalid_svg_shapes = ['rect', 'circle', 'ellipse', 'line', 'polyline', 'polygon']


def parse_single_path(path_str):
    ps = parse_path(path_str)
    # print(len(ps))

    stroke_points_list = []
    control_points_list = []
    for item_i, path_item in enumerate(ps):
        path_type = type(path_item)

        if path_type == path.Move:
            # assert item_i == 0
            if item_i != 0:
                assert (len(control_points_list) - 1) % 3 == 0
                assert len(control_points_list) > 1
                stroke_points_list.append(control_points_list)
                control_points_list = []

            start = path_item.start
            start_x, start_y = start.real, start.imag
            control_points_list.append((start_x, start_y))
        elif path_type == path.CubicBezier:
            start, control1, control2, end = path_item.start, path_item.control1, path_item.control2, path_item.end
            start_x, start_y = start.real, start.imag
            control1_x, control1_y = control1.real, control1.imag
            control2_x, control2_y = control2.real, control2.imag
            end_x, end_y = end.real, end.imag
            control_points_list.append((control1_x, control1_y))
            control_points_list.append((control2_x, control2_y))
            control_points_list.append((end_x, end_y))
        elif path_type == path.Arc:
            raise Exception('Arc is here')
        elif path_type == path.Line:
            # assert len(control_points_list) == 1
            # start, end = path_item.start, path_item.end
            # start_x, start_y = start.real, start.imag
            # end_x, end_y = end.real, end.imag
            #
            # control1_x = 2.0 / 3.0 * start_x + 1.0 / 3.0 * end_x
            # control1_y = 2.0 / 3.0 * start_y + 1.0 / 3.0 * end_y
            # control2_x = 1.0 / 3.0 * start_x + 2.0 / 3.0 * end_x
            # control2_y = 1.0 / 3.0 * start_y + 2.0 / 3.0 * end_y
            #
            # control1 = (control1_x, control1_y)
            # control2 = (control2_x, control2_y)
            # control1_dist = sample_random_position(control1, 4.0, 1.0)
            # control2_dist = sample_random_position(control2, 4.0, 1.0)
            #
            # control_points_list.append(control1_dist)
            # control_points_list.append(control2_dist)
            # control_points_list.append((end_x, end_y))
            raise Exception('Line is here')
        elif path_type == path.Close:
            assert item_i == len(ps) - 1
        else:
            raise Exception('Unknown path_type', path_type)

    assert (len(control_points_list) - 1) % 3 == 0
    assert len(control_points_list) > 1
    stroke_points_list.append(control_points_list)

    return stroke_points_list


def matrix_transform(points, matrix_params):
    # points: (N, 2), (x, y)
    # matrix_params: (6)
    new_points = []
    a, b, c, d, e, f = matrix_params
    matrix = np.array([[a, c, e],
                       [b, d, f],
                       [0, 0, 1]], dtype=np.float32)
    for point in points:
        point_vec = [point[0], point[1], 1]
        new_point = np.matmul(matrix, point_vec)[:2]
        new_points.append(new_point)
        # print(point, new_point)
    new_points = np.stack(new_points).astype(np.float32)
    return new_points


def parse_svg(svg_file):
    tree = ET.parse(svg_file)
    root = tree.getroot()

    width = root.get('width')
    height = root.get('height')
    width = int(width[:-2])
    height = int(height[:-2])

    view_box = root.get('viewBox')
    view_x, view_y, view_width, view_height = view_box.split(' ')
    view_x, view_y, view_width, view_height = int(view_x), int(view_y), int(view_width), int(view_height)
    assert view_x == 0 and view_y == 0
    assert width == view_width and height == view_height

    strokes_list = []

    for elem in root.iter():
        try:
            _, tag_suffix = elem.tag.split('}')
        except ValueError:
            continue

        assert tag_suffix not in invalid_svg_shapes

        if tag_suffix == 'path':
            path_d = elem.attrib['d']
            control_points_single_stroke_list = parse_single_path(path_d)  # (N_point, 2)
            for control_points_single_stroke_ in control_points_single_stroke_list:
                control_points_single_stroke = np.array(control_points_single_stroke_, dtype=np.float32)
                # print('control_points_single_stroke', control_points_single_stroke.shape)

                if 'transform' in elem.attrib.keys():
                    transformation = elem.attrib['transform']
                    if 'translate' in transformation:
                        translate_xy = transformation[transformation.find('(')+1:transformation.find(')')]
                        assert ', ' in translate_xy
                        translate_x = float(translate_xy[:translate_xy.find(', ')])
                        translate_y = float(translate_xy[translate_xy.find(', ')+2:])

                        control_points_single_stroke[:, 0] += translate_x
                        control_points_single_stroke[:, 1] += translate_y
                    elif 'matrix' in transformation:
                        matrix_params = transformation[transformation.find('(')+1:transformation.find(')')]
                        matrix_params = matrix_params.split(' ')
                        assert len(matrix_params) == 6
                        matrix_params = [float(item) for item in matrix_params]
                        control_points_single_stroke = matrix_transform(control_points_single_stroke, matrix_params)
                    else:
                        raise Exception('Error transformation')

                strokes_list.append(control_points_single_stroke)

    assert len(strokes_list) > 0
    return (view_width, view_height), strokes_list


def svg_to_npz(database, ref_image):
    svg_base = os.path.join(database, 'svg')
    img_base = os.path.join(database, 'raster')

    save_base_parameter = os.path.join(database, 'parameter')
    save_base_color_separate = os.path.join(database, 'vector_vis')
    os.makedirs(save_base_parameter, exist_ok=True)
    os.makedirs(save_base_color_separate, exist_ok=True)

    ref_id = ref_image[:ref_image.find('.')]
    svg_file_path = os.path.join(svg_base, ref_id + '.svg')
    img_file_path = os.path.join(img_base, ref_image)

    img = Image.open(img_file_path).convert('RGB')
    img_width, img_height = img.width, img.height
    assert img_width == img_height

    view_sizes, strokes_list = parse_svg(svg_file_path)
    # strokes_list: list of (N_point, 2), in view size
    assert view_sizes[0] == view_sizes[1]
    view_size = view_sizes[0]

    norm_strokes_list = []  # list of (N_point, 2), in image size
    max_seq_num = 0
    for single_stroke in strokes_list:
        single_stroke_norm = single_stroke / float(view_size) * float(img_width)
        norm_strokes_list.append(single_stroke_norm)
        segment_num = (len(single_stroke_norm) - 1) // 3
        # print('segment_num', segment_num)
        max_seq_num += segment_num
    print('max_seq_num', max_seq_num)

    save_npz_path = os.path.join(save_base_parameter, ref_id + '.npz')
    np.savez(save_npz_path, strokes_data=norm_strokes_list, canvas_size=img_width)

    # visualization
    pydiffvg.set_use_gpu(torch.cuda.is_available())

    background_ = torch.ones(img_width, img_width, 4)
    render_ = pydiffvg.RenderFunction.apply

    stroke_thickness = 1.2

    # black_stroke_img = draw_segment_separately(norm_strokes_list, img_width, stroke_thickness, max_seq_num,
    #                                            render_, background_, is_black=True)
    # black_stroke_img_save_path = os.path.join(img_base, ref_id + '-rendered.png')
    # pydiffvg.imwrite(black_stroke_img.cpu(), black_stroke_img_save_path, gamma=1.0)

    color_stroke_img = draw_segment_separately(norm_strokes_list, img_width, stroke_thickness, max_seq_num,
                                               render_, background_, is_black=False)
    color_stroke_img_save_path = os.path.join(save_base_color_separate, ref_id + '.png')
    pydiffvg.imwrite(color_stroke_img.cpu(), color_stroke_img_save_path, gamma=1.0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--database', '-db', type=str, default='../sample_inputs/rough/', help="define the data base")
    parser.add_argument('--reference', '-ref', type=str, default='20-0.png', help="define the reference image")
    args = parser.parse_args()

    svg_to_npz(args.database, args.reference)
