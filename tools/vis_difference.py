import os
import numpy as np
from PIL import Image
import argparse


def vis_difference(database_input, data_type, reference_image, target_image):
    data_base = os.path.join(database_input, data_type)
    raster_base = os.path.join(data_base, 'raster')
    raster_diff_base = os.path.join(data_base, 'raster_diff')
    os.makedirs(raster_diff_base, exist_ok=True)

    ref_img_path = os.path.join(raster_base, reference_image)
    tar_img_path = os.path.join(raster_base, target_image)

    ref_img = Image.open(ref_img_path).convert('L')
    ref_img = np.array(ref_img, dtype=np.float32)
    tar_img = Image.open(tar_img_path).convert('RGB')
    tar_img = np.array(tar_img, dtype=np.float32)

    target_mask = (tar_img < 200).any(-1)
    ref_img_bg = np.expand_dims(ref_img, axis=-1).astype(np.float32)
    ref_img_bg = np.concatenate(
        [np.ones_like(ref_img_bg) * 255,
         ref_img_bg,
         np.ones_like(ref_img_bg) * 255], axis=-1)
    ref_img_bg = 255 - (255 - ref_img_bg) * 0.5
    ref_img_bg[target_mask] = tar_img[target_mask]
    ref_img_bg = ref_img_bg.astype(np.uint8)

    save_path = os.path.join(raster_diff_base, reference_image[:-4] + '_and_' + target_image[:-4] + '.png')
    ref_img_bg_png = Image.fromarray(ref_img_bg, 'RGB')
    ref_img_bg_png.save(save_path, 'PNG')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--database_input', '-dbi', type=str, default='../sample_inputs',
                        help="define the input data base")
    parser.add_argument('--data_type', '-dt', type=str, default='rough', choices=['clean', 'rough'],
                        help="define the data type")
    parser.add_argument('--reference_image', '-ri', type=str, default='23-0.png',
                        help="define the reference image")
    parser.add_argument('--target_image', '-ti', type=str, default='23-1.png',
                        help="define the target image")

    args = parser.parse_args()

    vis_difference(args.database_input, args.data_type, args.reference_image, args.target_image)
