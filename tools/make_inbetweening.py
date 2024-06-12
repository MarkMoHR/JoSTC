import os
import numpy as np
from PIL import Image


def gen_intensity_list(max_inten, min_inten, num):
    interval = (max_inten - min_inten) // (num - 1)
    intensity_list = [min_inten + i * interval for i in range(num)]
    intensity_list = intensity_list[::-1]
    return intensity_list


def make_inbetweening_img(data_base, image_sequence):
    max_intensity = 200
    min_intensity = 0
    black_threshold = 128

    img_num = len(image_sequence)

    intensity_list = gen_intensity_list(max_intensity, min_intensity, img_num)
    print('intensity_list', intensity_list)

    img_inbetween = None

    for i, img_name in enumerate(image_sequence):
        img_path = os.path.join(data_base, img_name)
        img = Image.open(img_path).convert('RGB')
        img = np.array(img, dtype=np.uint8)[:, :, 0]

        intensity = intensity_list[i]

        if img_inbetween is None:
            img_inbetween = np.ones_like(img) * 255

        img_inbetween[img <= black_threshold] = intensity

    img_inbetween = Image.fromarray(img_inbetween, 'L')
    save_path = os.path.join(data_base, 'inbetweening.png')
    img_inbetween.save(save_path)


def make_inbetweening_gif(data_base, image_sequence):
    all_files = image_sequence + image_sequence[::-1][1:]

    gif_frames = []
    for img_name in all_files:
        img_i = Image.open(os.path.join(data_base, img_name))
        gif_frames.append(img_i)

    print('gif_frames', len(gif_frames))
    save_path = os.path.join(data_base, 'inbetweening.gif')
    first_frame = gif_frames[0]
    first_frame.save(save_path, save_all=True, append_images=gif_frames, loop=0, duration=0.01)


if __name__ == '__main__':
    data_base = '../outputs/inference/clean/inbetweening/9'
    image_sequence = ['10.png', '18.png', '25.png', '32.png', '40.png']

    make_inbetweening_img(data_base, image_sequence)
    make_inbetweening_gif(data_base, image_sequence)
