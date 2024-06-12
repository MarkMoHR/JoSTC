import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


class VGG_Slim(nn.Module):
    def __init__(self):
        super(VGG_Slim, self).__init__()

        self.conv11 = nn.Conv2d(1, 64, 3, 1, 1)
        self.conv12 = nn.Conv2d(64, 64, 3, 1, 1)

        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv21 =nn.Conv2d(64, 128, 3, 1, 1)
        self.conv22 =nn.Conv2d(128, 128, 3, 1, 1)

        self.conv31 =nn.Conv2d(128, 256, 3, 1, 1)
        self.conv32 =nn.Conv2d(256, 256, 3, 1, 1)
        self.conv33 =nn.Conv2d(256, 256, 3, 1, 1)

        self.conv41 =nn.Conv2d(256, 512, 3, 1, 1)
        self.conv42 =nn.Conv2d(512, 512, 3, 1, 1)
        self.conv43 =nn.Conv2d(512, 512, 3, 1, 1)

        self.conv51 =nn.Conv2d(512, 512, 3, 1, 1)
        self.conv52 =nn.Conv2d(512, 512, 3, 1, 1)
        self.conv53 =nn.Conv2d(512, 512, 3, 1, 1)

    def forward(self, input_imgs):
        return_map = {}

        x = input_imgs
        if x.dim() == 3:
            x = x.unsqueeze(dim=1)  # NCHW

        x = F.relu(self.conv11(x))
        return_map['ReLU1_1'] = x
        x = F.relu(self.conv12(x))
        return_map['ReLU1_2'] = x
        x = self.max_pool(x)

        x = F.relu(self.conv21(x))
        return_map['ReLU2_1'] = x
        x = F.relu(self.conv22(x))
        return_map['ReLU2_2'] = x
        x = self.max_pool(x)

        x = F.relu(self.conv31(x))
        return_map['ReLU3_1'] = x
        x = F.relu(self.conv32(x))
        return_map['ReLU3_2'] = x
        x = F.relu(self.conv33(x))
        return_map['ReLU3_3'] = x
        x = self.max_pool(x)

        x = F.relu(self.conv41(x))
        return_map['ReLU4_1'] = x
        x = F.relu(self.conv42(x))
        return_map['ReLU4_2'] = x
        x = F.relu(self.conv43(x))
        return_map['ReLU4_3'] = x
        x = self.max_pool(x)

        x = F.relu(self.conv51(x))
        return_map['ReLU5_1'] = x
        x = F.relu(self.conv52(x))
        return_map['ReLU5_2'] = x
        x = F.relu(self.conv53(x))
        return_map['ReLU5_3'] = x

        return return_map


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import os
    from PIL import Image

    os.environ['CUDA_VISIBLE_DEVICES'] = '2'

    use_cuda = torch.cuda.is_available()

    def load_image(img_path):
        img = Image.open(img_path).convert('RGB')
        img = np.array(img, dtype=np.float32)[:, :, 0] / 255.0
        return img

    raster_size = 128
    vis_layer1 = 'ReLU3_3'
    vis_layer2 = 'ReLU5_1'

    vgg_slim_model = VGG_Slim()

    model_path = '../models/quickdraw-perceptual.pth'
    pretrained_dict = torch.load(model_path)
    print('pretrained_dict')
    print(pretrained_dict.keys())

    model_dict = vgg_slim_model.state_dict()
    print('model_dict')
    print(model_dict.keys())

    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    vgg_slim_model.load_state_dict(model_dict)

    vgg_slim_model.eval()

    param_list = vgg_slim_model.named_parameters()
    print('-' * 100)
    count_t_vars = 0
    for name, param in param_list:
        num_param = np.prod(list(param.size()))
        count_t_vars += num_param
        print('%s | shape: %s | num_param: %i' % (name, str(param.size()), num_param))
    print('Total trainable variables %i.' % count_t_vars)
    print('-' * 100)

    image_types = ['GT', 'Connected', 'Disconnected', 'Parallel', 'Part']
    image_info_map = {}

    all_files = os.listdir(os.path.join('../vgg_utils/testData', 'GT'))
    all_files.sort()
    img_ids = []
    for file_name in all_files:
        img_idx = file_name[:file_name.find('_')]
        img_ids.append(img_idx)
    print(img_ids)

    if use_cuda:
        vgg_slim_model = vgg_slim_model.cuda()

    for img_id in img_ids:
        for image_type in image_types:
            finalstr = '_gt.png' if image_type == 'GT' else '_pred.png'
            img_path = os.path.join('../vgg_utils/testData', image_type, str(img_id) + finalstr)
            print(img_path)
            image = load_image(img_path)
            image_tensor = torch.tensor(np.expand_dims(image, axis=0)).float()

            if use_cuda:
                image_tensor = image_tensor.cuda()

            feature_maps = vgg_slim_model(image_tensor)
            feature_maps_11 = feature_maps['ReLU3_3'].cpu().data.numpy()
            feature_maps_33 = feature_maps['ReLU5_1'].cpu().data.numpy()

            print('ReLU3_3', feature_maps_11.shape)
            print('ReLU5_1', feature_maps_33.shape)

            feature_maps_val11 = np.transpose(feature_maps_11[0], (1, 2, 0))
            feature_maps_val33 = np.transpose(feature_maps_33[0], (1, 2, 0))

            if image_type != 'GT':
                feature_maps_gt11 = image_info_map['GT'][4]
                feat_diff_all11 = np.mean(np.abs(feature_maps_gt11 - feature_maps_val11), axis=-1)
                perc_layer_loss11 = np.mean(np.abs(feature_maps_gt11 - feature_maps_val11))

                feature_maps_gt33 = image_info_map['GT'][1]
                feat_diff_all33 = np.mean(np.abs(feature_maps_gt33 - feature_maps_val33), axis=-1)
                perc_layer_loss33 = np.mean(np.abs(feature_maps_gt33 - feature_maps_val33))
                # print('perc_layer_loss', image_type, perc_layer_loss)
            else:
                feat_diff_all11 = np.zeros_like(feature_maps_val11[:, :, 0])
                perc_layer_loss11 = 0.0

                feat_diff_all33 = np.zeros_like(feature_maps_val33[:, :, 0])
                perc_layer_loss33 = 0.0

            image_info_map[image_type] = [image,
                                          feature_maps_val33, feat_diff_all33, perc_layer_loss33,
                                          feature_maps_val11, feat_diff_all11, perc_layer_loss11]

        rows = 3
        cols = len(image_types)
        plt.figure(figsize=(4 * cols, 4 * rows))

        for image_type_i, image_type in enumerate(image_types):
            input_image = image_info_map[image_type][0]
            feat_diff33 = image_info_map[image_type][2]
            perc_loss33 = image_info_map[image_type][3]
            feat_diff11 = image_info_map[image_type][5]
            perc_loss11 = image_info_map[image_type][6]

            plt.subplot(rows, cols, image_type_i + 1)
            plt.title(image_type, fontsize=12)
            if image_type_i != 0:
                plt.axis('off')
            plt.imshow(input_image)

            plt.subplot(rows, cols, image_type_i + 1 + len(image_types))
            plt.title(str(perc_loss11), fontsize=12)
            if image_type_i != 0:
                plt.axis('off')
            plt.imshow(feat_diff11)

            plt.subplot(rows, cols, image_type_i + 1 + len(image_types) + len(image_types))
            plt.title(str(perc_loss33), fontsize=12)
            if image_type_i != 0:
                plt.axis('off')
            plt.imshow(feat_diff33)

        plt.show()
