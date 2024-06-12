import os
import numpy as np
from PIL import Image
import random
import argparse

from xml.dom import minidom


def write_svg(points_list, height, width, svg_save_path):
    impl_save = minidom.getDOMImplementation()

    doc_save = impl_save.createDocument(None, None, None)

    rootElement_save = doc_save.createElement('svg')
    rootElement_save.setAttribute('xmlns', 'http://www.w3.org/2000/svg')

    rootElement_save.setAttribute('height', str(height) + 'pt')
    rootElement_save.setAttribute('width', str(width) + 'pt')

    view_box = '0 0 ' + str(width) + ' ' + str(height)
    rootElement_save.setAttribute('viewBox', view_box)

    globl_path_i = 0
    for stroke_i, stroke_points in enumerate(points_list):
        # stroke_points: (N_point, 2), in image size
        segment_num = (stroke_points.shape[0] - 1) // 3

        for segment_i in range(segment_num):
            start_idx = segment_i * 3
            start_point = stroke_points[start_idx]
            ctrl_point1 = stroke_points[start_idx + 1]
            ctrl_point2 = stroke_points[start_idx + 2]
            end_point = stroke_points[start_idx + 3]

            command_str = 'M ' + str(start_point[0]) + ', ' + str(start_point[1]) + ' '
            command_str += 'C ' + str(ctrl_point1[0]) + ', ' + str(ctrl_point1[1]) + ' ' \
                           + str(ctrl_point2[0]) + ', ' + str(ctrl_point2[1]) + ' ' \
                           + str(end_point[0]) + ', ' + str(end_point[1]) + ' '

            childElement_save = doc_save.createElement('path')
            childElement_save.setAttribute('id', 'curve_' + str(globl_path_i))
            childElement_save.setAttribute('stroke', '#000000')
            childElement_save.setAttribute('stroke-linejoin', 'round')
            childElement_save.setAttribute('stroke-linecap', 'square')
            childElement_save.setAttribute('fill', 'none')

            childElement_save.setAttribute('d', command_str)
            childElement_save.setAttribute('stroke-width', str(2.375))
            rootElement_save.appendChild(childElement_save)

            globl_path_i += 1

    doc_save.appendChild(rootElement_save)

    f = open(svg_save_path, 'w')
    doc_save.writexml(f, addindent='  ', newl='\n')
    f.close()


def write_svg_chain(points_list, height, width, svg_save_path):
    impl_save = minidom.getDOMImplementation()

    doc_save = impl_save.createDocument(None, None, None)

    rootElement_save = doc_save.createElement('svg')
    rootElement_save.setAttribute('xmlns', 'http://www.w3.org/2000/svg')

    rootElement_save.setAttribute('height', str(height) + 'pt')
    rootElement_save.setAttribute('width', str(width) + 'pt')

    view_box = '0 0 ' + str(width) + ' ' + str(height)
    rootElement_save.setAttribute('viewBox', view_box)

    for stroke_i, stroke_points in enumerate(points_list):
        # stroke_points: (N_point, 2), in image size
        segment_num = (stroke_points.shape[0] - 1) // 3

        command_str = 'M ' + str(stroke_points[0][0]) + ', ' + str(stroke_points[0][1]) + ' '

        for segment_i in range(segment_num):
            start_idx = segment_i * 3
            ctrl_point1 = stroke_points[start_idx + 1]
            ctrl_point2 = stroke_points[start_idx + 2]
            end_point = stroke_points[start_idx + 3]

            command_str += 'C ' + str(ctrl_point1[0]) + ', ' + str(ctrl_point1[1]) + ' ' \
                           + str(ctrl_point2[0]) + ', ' + str(ctrl_point2[1]) + ' ' \
                           + str(end_point[0]) + ', ' + str(end_point[1]) + ' '

        childElement_save = doc_save.createElement('path')
        childElement_save.setAttribute('id', 'curve_' + str(stroke_i))
        childElement_save.setAttribute('stroke', '#000000')
        childElement_save.setAttribute('stroke-linejoin', 'round')
        childElement_save.setAttribute('stroke-linecap', 'square')
        childElement_save.setAttribute('fill', 'none')

        childElement_save.setAttribute('d', command_str)
        childElement_save.setAttribute('stroke-width', str(2.375))
        rootElement_save.appendChild(childElement_save)

    doc_save.appendChild(rootElement_save)

    f = open(svg_save_path, 'w')
    doc_save.writexml(f, addindent='  ', newl='\n')
    f.close()


def npz_to_svg(file_names, database_input, database_output, data_type):
    svg_chain_save_base = os.path.join(database_output, data_type, 'svg', 'chain')
    svg_separate_save_base = os.path.join(database_output, data_type, 'svg', 'separate')
    os.makedirs(svg_chain_save_base, exist_ok=True)
    os.makedirs(svg_separate_save_base, exist_ok=True)

    for file_name in file_names:
        file_id = file_name[:file_name.find('.')]

        output_image_path = os.path.join(database_output, data_type, 'raster', file_name)
        output_image = Image.open(output_image_path)
        width, height = output_image.width, output_image.height

        # output
        npz_path = os.path.join(database_output, data_type, 'parameter', file_id + '.npz')
        npz = np.load(npz_path, encoding='latin1', allow_pickle=True)
        strokes_data_output = npz['strokes_data']  # list of (N_point, 2), in image size

        svg_chain_output_save_path = os.path.join(svg_chain_save_base, file_id + '.svg')
        write_svg_chain(strokes_data_output, height, width, svg_chain_output_save_path)
        svg_separate_output_save_path = os.path.join(svg_separate_save_base, file_id + '.svg')
        write_svg(strokes_data_output, height, width, svg_separate_output_save_path)

        # rewrite input
        file_id = file_name[:file_name.find('-')]

        npz_path = os.path.join(database_input, data_type, 'parameter', file_id + '-0.npz')
        npz = np.load(npz_path, encoding='latin1', allow_pickle=True)
        strokes_data_input = npz['strokes_data']  # list of (N_point, 2), in image size

        svg_chain_input_save_path = os.path.join(svg_chain_save_base, file_id + '-0.svg')
        write_svg_chain(strokes_data_input, height, width, svg_chain_input_save_path)
        svg_chain_input_save_path = os.path.join(svg_separate_save_base, file_id + '-0.svg')
        write_svg(strokes_data_input, height, width, svg_chain_input_save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--database_input', '-dbi', type=str, default='../sample_inputs', help="define the input data base")
    parser.add_argument('--database_output', '-dbo', type=str, default='../outputs/inference', help="define the output data base")
    parser.add_argument('--data_type', '-dt', type=str, default='rough', choices=['clean', 'rough'], help="define the data type")
    parser.add_argument('--file_names', '-fn', type=str, default=['23-1.png', '23-2.png'], nargs='+', help="define the file names")

    args = parser.parse_args()

    npz_to_svg(args.file_names, args.database_input, args.database_output, args.data_type)
