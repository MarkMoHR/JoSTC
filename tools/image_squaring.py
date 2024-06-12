from PIL import Image
import numpy as np
import argparse


def image_squaring(img_path):
    img = Image.open(img_path).convert('RGB')
    height, width = img.height, img.width

    max_dim = max(height, width)

    pad_top = (max_dim - height) // 2
    pad_down = max_dim - height - pad_top
    pad_left = (max_dim - width) // 2
    pad_right = max_dim - width - pad_left

    img = np.array(img, dtype=np.uint8)
    img_p = np.pad(img, ((pad_top, pad_down), (pad_left, pad_right), (0, 0)), 'constant', constant_values=255)
    img_p = Image.fromarray(img_p, 'RGB')
    img_p.save(img_path[:-4] + '-pad.png', 'PNG')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', '-f', type=str, default='', help="define an image")
    args = parser.parse_args()

    image_squaring(args.file)
