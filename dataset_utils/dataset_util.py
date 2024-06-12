import pydiffvg
import torch

from dataset_utils.common import generate_colors2, generate_colors_ordered


def draw_segment_jointly(strokes_points_list, canvas_size, stroke_thickness, render, background, is_black=True):
    shapes = []
    shape_groups = []

    stroke_num = len(strokes_points_list)
    colors = generate_colors2(stroke_num)  # list of (3), in [0., 1.]

    for i in range(stroke_num):
        points_single_stroke = strokes_points_list[i]  # (N, 2)
        points_single_stroke = torch.tensor(points_single_stroke)

        segment_num = (len(points_single_stroke) - 1) // 3
        if is_black:
            stroke_color = [0.0, 0.0, 0.0, 1.0]
        else:
            stroke_color = [colors[i][0], colors[i][1], colors[i][2], 1.0]

        num_control_points = torch.zeros(segment_num, dtype=torch.int32) + 2

        path = pydiffvg.Path(num_control_points=num_control_points,
                             points=points_single_stroke,
                             is_closed=False,
                             stroke_width=torch.tensor(stroke_thickness))
        shapes.append(path)

        path_group = pydiffvg.ShapeGroup(shape_ids=torch.tensor([len(shapes) - 1]),
                                         fill_color=None,
                                         stroke_color=torch.tensor(stroke_color))
        shape_groups.append(path_group)

    scene_args_t = pydiffvg.RenderFunction.serialize_scene(
        canvas_size, canvas_size, shapes, shape_groups)
    img = render(canvas_size,  # width
                 canvas_size,  # height
                 2,  # num_samples_x
                 2,  # num_samples_y
                 1,  # seed
                 background,  # background_image
                 *scene_args_t)
    return img


def draw_segment_separately(strokes_points_list, canvas_size, stroke_thickness, max_seq_number,
                            render, background, is_black=False, draw_order=False):
    shapes = []
    shape_groups = []

    stroke_num = len(strokes_points_list)

    if draw_order:
        colors = generate_colors_ordered(max_seq_number)  # list of (3), in [0., 1.]
    else:
        colors = generate_colors2(max_seq_number)  # list of (3), in [0., 1.]
    global_segment_idx = 0

    for i in range(stroke_num):
        points_single_stroke = strokes_points_list[i]  # (N, 2)
        points_single_stroke = torch.tensor(points_single_stroke)

        segment_num = (len(points_single_stroke) - 1) // 3

        for j in range(segment_num):
            start_idx = j * 3
            points_single_segment = points_single_stroke[start_idx: start_idx + 4]

            if is_black:
                stroke_color = [0.0, 0.0, 0.0, 1.0]
            else:
                stroke_color = [colors[global_segment_idx][0], colors[global_segment_idx][1], colors[global_segment_idx][2], 1.0]

            num_control_points = torch.tensor([2])

            path = pydiffvg.Path(num_control_points=num_control_points,
                                 points=points_single_segment,
                                 is_closed=False,
                                 stroke_width=torch.tensor(stroke_thickness))
            shapes.append(path)

            path_group = pydiffvg.ShapeGroup(shape_ids=torch.tensor([len(shapes) - 1]),
                                             fill_color=None,
                                             stroke_color=torch.tensor(stroke_color))
            shape_groups.append(path_group)

            global_segment_idx += 1

    scene_args_t = pydiffvg.RenderFunction.serialize_scene(
        canvas_size, canvas_size, shapes, shape_groups)
    img = render(canvas_size,  # width
                 canvas_size,  # height
                 2,  # num_samples_x
                 2,  # num_samples_y
                 1,  # seed
                 background,  # background_image
                 *scene_args_t)
    return img
