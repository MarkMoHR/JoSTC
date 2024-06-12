import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import time
from tqdm import tqdm
import os
from PIL import Image
from collections import OrderedDict

from vgg_utils.VGG16 import VGG_Slim
from dataset_utils.common import generate_colors2
from hparam import HParams
from rnn2 import HyperLSTMCell
from utils import correspondence_clinging, get_correspondence_window_size, spatial_transform_reverse_point, \
    image_cropping_stn, normalize_image_m1to1, draw_dot, seq_params_to_list

import pydiffvg

pydiffvg.set_use_gpu(torch.cuda.is_available())
print('Setting pydiffvg.set_use_gpu:', torch.cuda.is_available())


def get_default_hparams():
    """Return default HParams for sketch-rnn."""
    hparams = HParams(
        add_coordconv=True,
        use_atrous_conv=False,
        first_kernel_size=3,  # 7 or 3

        z_size=256,  # Size of latent vector z.

        # parameters for two transformation modules
        use_square_window=False,
        init_window_size_corres_trans=0.6,

        transform_with_rotation=True,
        transform_use_global_info=True,
        enc_model_transform='combined',  # ['combined', 'separated']
        dec_model_transform='mlp',  # ['rnn', 'mlp']
        transform_module_zero_init='last',  # ['none', 'last', 'all']
        frozen_transform_module=True,

        # parameters for correspondence module
        enc_model_correspondence='separated',  # ['combined', 'separated']
        raster_size_corres=256,  # cropping size for starting point correspondence module
        use_clinging=True,
        clinging_binary_threshold=128.0,

        use_segment_img=True,
        use_reference_canvas=False,
        use_target_canvas=True,

        use_attn_corres=True,
        attn_type_corres='SA',
        sa_block_pos_corres=3,  # [1, 2, 3, 4]

        use_dropout=False,
        dropout_rate=0.3,  # probability of an element to be zeroed

        # parameters for tracing module
        raster_size=192,

        window_size_scaling_ref=1.5,  # [1.25, 1.5, 2.0]
        window_size_scaling_init_tar=1.5,  # [1.25, 1.5, 2.0]
        window_size_scaling_times_tar=(0.2, 2.0),
        window_size_min=48,  # [1.25, 1.5, 2.0]

        hidden_states_zero=True,  # whether setting input hidden states to zero for starting of each stroke

        enc_model_tracing='separated',  # ['combined', 'separated']
        dec_model_tracing='rnn',  # ['rnn', 'mlp']
        dec_rnn_size=256,  # Size of decoder.
        rnn_model='hyper',  # Decoder: lstm, layer_norm or hyper.

        stroke_thickness=1.2,  # 2.0 for toy; 1.2 for TUB

        raster_loss_base_type='perceptual',  # [l1, mse, perceptual]
        perc_loss_layers=['ReLU1_2', 'ReLU2_2', 'ReLU3_3', 'ReLU4_3', 'ReLU5_1'],
        perc_loss_fuse_type='add',  # ['max', 'add', 'raw_add', 'weighted_sum']
        perceptual_model_path='models/quickdraw-perceptual.pth',

        trained_models_dir='models',
        inference_root='outputs/inference'
    )
    return hparams


def general_conv2d(in_dim, output_dim, kernel_size, stride, do_norm=True, norm_type='instance_norm', padding=1,
                   atrous=False, atrous_rate=1):
    if atrous:
        conv = nn.Conv2d(in_dim, output_dim, kernel_size=kernel_size, stride=stride, padding=atrous_rate, dilation=atrous_rate)
    else:
        conv = nn.Conv2d(in_dim, output_dim, kernel_size=kernel_size, stride=stride, padding=padding)

    if do_norm:
        if norm_type == 'instance_norm':
            norm = nn.InstanceNorm2d(output_dim, affine=True)
        elif norm_type == 'batch_norm':
            norm = nn.BatchNorm2d(output_dim)
        elif norm_type == 'layer_norm':
            norm = nn.LayerNorm(output_dim)
        else:
            raise Exception('Unknown norm_type:', norm_type)

        return nn.Sequential(
            OrderedDict([
                ('conv', conv),
                (norm_type, norm)
            ]))
    else:
        return conv


class SelfAttention(nn.Module):
    def __init__(self, in_dim):
        super(SelfAttention, self).__init__()        
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.output_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        inputs :
            x : input feature maps (N, C, h, w)
        returns :
            out : self attention value + input feature
            attention: N X hw X hw
        """
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, height * width).permute(0, 2, 1)  # (N, hw, c)
        proj_key = self.key_conv(x).view(m_batchsize, -1, height * width)  # (N, c, hw)
        energy = torch.bmm(proj_query, proj_key)  # (N, hw, hw)
        attn_map = self.softmax(energy)  # (N, hw, hw)
        proj_value = self.value_conv(x).view(m_batchsize, -1, height * width)  # (N, C, hw)

        x_attn = torch.bmm(proj_value, attn_map.permute(0, 2, 1))  # (N, C, hw)
        x_attn = x_attn.view(m_batchsize, C, height, width)  # (N, C, h, w)
        x_attn = self.output_conv(x_attn)  # (N, C, h, w)
        
        out = self.gamma * x_attn + x
        return out, attn_map


class CNN_SepEncoder_correspondence(nn.Module):
    def __init__(self, input_dim_ref, input_dim_tar, output_dim, input_size, use_atrous,
                 use_attn, attn_type=None, sa_block_pos=None, use_dropout=False, dropout_rate=0.0):
        super(CNN_SepEncoder_correspondence, self).__init__()

        if use_atrous:
            atrou_rates = [1, 1, 2, 4, 4]
        else:
            atrou_rates = [1, 1, 1, 1, 1]

        # reference
        self.cnn_enc_11_ref = general_conv2d(input_dim_ref, 16, kernel_size=3, stride=1, atrous=use_atrous, atrous_rate=atrou_rates[0])
        self.cnn_enc_12_ref = general_conv2d(16, 32, kernel_size=3, stride=2, atrous=use_atrous, atrous_rate=atrou_rates[0])

        self.cnn_enc_21_ref = general_conv2d(32, 32, kernel_size=3, stride=1, atrous=use_atrous, atrous_rate=atrou_rates[1])
        self.cnn_enc_22_ref = general_conv2d(32, 64, kernel_size=3, stride=2, atrous=use_atrous, atrous_rate=atrou_rates[1])

        self.cnn_enc_31_ref = general_conv2d(64, 64, kernel_size=3, stride=1, atrous=use_atrous, atrous_rate=atrou_rates[2])
        self.cnn_enc_32_ref = general_conv2d(64, 64, kernel_size=3, stride=1, atrous=use_atrous, atrous_rate=atrou_rates[2])
        self.cnn_enc_33_ref = general_conv2d(64, 128, kernel_size=3, stride=2, atrous=use_atrous, atrous_rate=atrou_rates[2])

        self.cnn_enc_41_ref = general_conv2d(128, 128, kernel_size=3, stride=1, atrous=use_atrous, atrous_rate=atrou_rates[3])
        self.cnn_enc_42_ref = general_conv2d(128, 128, kernel_size=3, stride=1, atrous=use_atrous, atrous_rate=atrou_rates[3])
        self.cnn_enc_43_ref = general_conv2d(128, 256, kernel_size=3, stride=2, atrous=use_atrous, atrous_rate=atrou_rates[3])

        # target
        self.cnn_enc_11_tar = general_conv2d(input_dim_tar, 16, kernel_size=3, stride=1, atrous=use_atrous, atrous_rate=atrou_rates[0])
        self.cnn_enc_12_tar = general_conv2d(16, 32, kernel_size=3, stride=2, atrous=use_atrous, atrous_rate=atrou_rates[0])

        self.cnn_enc_21_tar = general_conv2d(64, 32, kernel_size=3, stride=1, atrous=use_atrous, atrous_rate=atrou_rates[1])
        self.cnn_enc_22_tar = general_conv2d(32, 64, kernel_size=3, stride=2, atrous=use_atrous, atrous_rate=atrou_rates[1])

        self.cnn_enc_31_tar = general_conv2d(128, 64, kernel_size=3, stride=1, atrous=use_atrous, atrous_rate=atrou_rates[2])
        self.cnn_enc_32_tar = general_conv2d(64, 64, kernel_size=3, stride=1, atrous=use_atrous, atrous_rate=atrou_rates[2])
        self.cnn_enc_33_tar = general_conv2d(64, 128, kernel_size=3, stride=2, atrous=use_atrous, atrous_rate=atrou_rates[2])

        self.cnn_enc_41_tar = general_conv2d(256, 128, kernel_size=3, stride=1, atrous=use_atrous, atrous_rate=atrou_rates[3])
        self.cnn_enc_42_tar = general_conv2d(128, 128, kernel_size=3, stride=1, atrous=use_atrous, atrous_rate=atrou_rates[3])
        self.cnn_enc_43_tar = general_conv2d(128, 256, kernel_size=3, stride=2, atrous=use_atrous, atrous_rate=atrou_rates[3])

        self.cnn_enc_51_tar = general_conv2d(512, 256, kernel_size=3, stride=1, atrous=use_atrous, atrous_rate=atrou_rates[4])
        self.cnn_enc_52_tar = general_conv2d(256, 256, kernel_size=3, stride=1, atrous=use_atrous, atrous_rate=atrou_rates[4])
        self.cnn_enc_53_tar = general_conv2d(256, 512, kernel_size=3, stride=2, atrous=use_atrous, atrous_rate=atrou_rates[4])

        self.use_attn = [False for _ in range(4)]
        if use_attn:
            if attn_type == 'SA':
                assert sa_block_pos in [1, 2, 3, 4]
                self.attn_1 = SelfAttention(in_dim=64) if sa_block_pos == 1 else None
                self.attn_2 = SelfAttention(in_dim=128) if sa_block_pos == 2 else None
                self.attn_3 = SelfAttention(in_dim=256) if sa_block_pos == 3 else None
                self.attn_4 = SelfAttention(in_dim=512) if sa_block_pos == 4 else None
                self.use_attn[int(sa_block_pos - 1)] = True
            else:
                raise Exception('Unknown attn_type:', attn_type)

        assert input_size % 32 == 0
        self.feature_size = input_size // 32
        self.gap = nn.AvgPool2d(self.feature_size)

        self.use_dropout = use_dropout
        if self.use_dropout:
            self.dropout = nn.Dropout(dropout_rate)

        self.fc = nn.Linear(512, output_dim)

    def forward(self, inputs_ref, inputs_tar):
        x_r = inputs_ref

        x_r11 = F.relu(self.cnn_enc_11_ref(x_r))
        x_r12 = F.relu(self.cnn_enc_12_ref(x_r11))

        x_r21 = F.relu(self.cnn_enc_21_ref(x_r12))
        x_r22 = F.relu(self.cnn_enc_22_ref(x_r21))

        x_r31 = F.relu(self.cnn_enc_31_ref(x_r22))
        x_r32 = F.relu(self.cnn_enc_32_ref(x_r31))
        x_r33 = F.relu(self.cnn_enc_33_ref(x_r32))

        x_r41 = F.relu(self.cnn_enc_41_ref(x_r33))
        x_r42 = F.relu(self.cnn_enc_42_ref(x_r41))
        x_r43 = F.relu(self.cnn_enc_43_ref(x_r42))

        x = inputs_tar

        x = F.relu(self.cnn_enc_11_tar(x))
        x = F.relu(self.cnn_enc_12_tar(x))

        x = torch.cat([x, x_r12], dim=1)
        if self.use_attn[0]:
            x = F.relu(self.attn_1(x)[0])

        x = F.relu(self.cnn_enc_21_tar(x))
        x = F.relu(self.cnn_enc_22_tar(x))

        x = torch.cat([x, x_r22], dim=1)
        if self.use_attn[1]:
            x = F.relu(self.attn_2(x)[0])

        x = F.relu(self.cnn_enc_31_tar(x))
        x = F.relu(self.cnn_enc_32_tar(x))
        x = F.relu(self.cnn_enc_33_tar(x))

        x = torch.cat([x, x_r33], dim=1)
        if self.use_attn[2]:
            x = F.relu(self.attn_3(x)[0])

        x = F.relu(self.cnn_enc_41_tar(x))
        x = F.relu(self.cnn_enc_42_tar(x))
        x = F.relu(self.cnn_enc_43_tar(x))

        x = torch.cat([x, x_r43], dim=1)
        if self.use_attn[3]:
            x = F.relu(self.attn_4(x)[0])

        x = F.relu(self.cnn_enc_51_tar(x))
        x = F.relu(self.cnn_enc_52_tar(x))
        x = F.relu(self.cnn_enc_53_tar(x))  # (N, C, H/32, W/32)

        x = self.gap(x)  # (N, C, 1, 1)
        x = torch.reshape(x, (x.size(0), -1))  # (N, C)

        if self.use_dropout:
            x = self.dropout(x)

        x = self.fc(x)  # (N, 2)
        x = torch.tanh(x)  # (N, 2), [-1.0, 1.0]

        return x


class CNN_SepEncoder(nn.Module):
    def __init__(self, input_dim_ref, input_dim_tar, output_dim, input_size, first_kernel_size, first_padding, use_atrous):
        super(CNN_SepEncoder, self).__init__()

        if use_atrous:
            atrou_rates = [1, 1, 2, 4, 4]
        else:
            atrou_rates = [1, 1, 1, 1, 1]

        # reference
        self.cnn_enc_11_ref = general_conv2d(input_dim_ref, 32, kernel_size=first_kernel_size, stride=2, padding=first_padding, atrous=use_atrous, atrous_rate=atrou_rates[0])
        self.cnn_enc_12_ref = general_conv2d(32, 32, kernel_size=3, stride=1, atrous=use_atrous, atrous_rate=atrou_rates[0])

        self.cnn_enc_21_ref = general_conv2d(32, 64, kernel_size=3, stride=2, atrous=use_atrous, atrous_rate=atrou_rates[1])
        self.cnn_enc_22_ref = general_conv2d(64, 64, kernel_size=3, stride=1, atrous=use_atrous, atrous_rate=atrou_rates[1])

        self.cnn_enc_31_ref = general_conv2d(64, 128, kernel_size=3, stride=2, atrous=use_atrous, atrous_rate=atrou_rates[2])
        self.cnn_enc_32_ref = general_conv2d(128, 128, kernel_size=3, stride=1, atrous=use_atrous, atrous_rate=atrou_rates[2])
        self.cnn_enc_33_ref = general_conv2d(128, 128, kernel_size=3, stride=1, atrous=use_atrous, atrous_rate=atrou_rates[2])

        self.cnn_enc_41_ref = general_conv2d(128, 256, kernel_size=3, stride=2, atrous=use_atrous, atrous_rate=atrou_rates[3])
        self.cnn_enc_42_ref = general_conv2d(256, 256, kernel_size=3, stride=1, atrous=use_atrous, atrous_rate=atrou_rates[3])
        self.cnn_enc_43_ref = general_conv2d(256, 256, kernel_size=3, stride=1, atrous=use_atrous, atrous_rate=atrou_rates[3])

        # target
        self.cnn_enc_11_tar = general_conv2d(input_dim_tar, 32, kernel_size=first_kernel_size, stride=2, padding=first_padding, atrous=use_atrous, atrous_rate=atrou_rates[0])
        self.cnn_enc_12_tar = general_conv2d(32, 32, kernel_size=3, stride=1, atrous=use_atrous, atrous_rate=atrou_rates[0])

        self.cnn_enc_21_tar = general_conv2d(64, 64, kernel_size=3, stride=2, atrous=use_atrous, atrous_rate=atrou_rates[1])
        self.cnn_enc_22_tar = general_conv2d(64, 64, kernel_size=3, stride=1, atrous=use_atrous, atrous_rate=atrou_rates[1])

        self.cnn_enc_31_tar = general_conv2d(128, 128, kernel_size=3, stride=2, atrous=use_atrous, atrous_rate=atrou_rates[2])
        self.cnn_enc_32_tar = general_conv2d(128, 128, kernel_size=3, stride=1, atrous=use_atrous, atrous_rate=atrou_rates[2])
        self.cnn_enc_33_tar = general_conv2d(128, 128, kernel_size=3, stride=1, atrous=use_atrous, atrous_rate=atrou_rates[2])

        self.cnn_enc_41_tar = general_conv2d(256, 256, kernel_size=3, stride=2, atrous=use_atrous, atrous_rate=atrou_rates[3])
        self.cnn_enc_42_tar = general_conv2d(256, 256, kernel_size=3, stride=1, atrous=use_atrous, atrous_rate=atrou_rates[3])
        self.cnn_enc_43_tar = general_conv2d(256, 256, kernel_size=3, stride=1, atrous=use_atrous, atrous_rate=atrou_rates[3])

        self.cnn_enc_51_tar = general_conv2d(512, 512, kernel_size=3, stride=2, atrous=use_atrous, atrous_rate=atrou_rates[4])
        self.cnn_enc_52_tar = general_conv2d(512, 512, kernel_size=3, stride=1, atrous=use_atrous, atrous_rate=atrou_rates[4])
        self.cnn_enc_53_tar = general_conv2d(512, 512, kernel_size=3, stride=1, atrous=use_atrous, atrous_rate=atrou_rates[4])

        assert input_size % 32 == 0
        self.feature_size = input_size // 32
        self.fc = nn.Linear(512 * self.feature_size * self.feature_size, output_dim)

        self.use_attn = [False for _ in range(4)]

    def forward(self, inputs_ref, inputs_tar):
        x_r = inputs_ref

        x_r11 = F.relu(self.cnn_enc_11_ref(x_r))
        x_r12 = F.relu(self.cnn_enc_12_ref(x_r11))

        x_r21 = F.relu(self.cnn_enc_21_ref(x_r12))
        x_r22 = F.relu(self.cnn_enc_22_ref(x_r21))

        x_r31 = F.relu(self.cnn_enc_31_ref(x_r22))
        x_r32 = F.relu(self.cnn_enc_32_ref(x_r31))
        x_r33 = F.relu(self.cnn_enc_33_ref(x_r32))

        x_r41 = F.relu(self.cnn_enc_41_ref(x_r33))
        x_r42 = F.relu(self.cnn_enc_42_ref(x_r41))
        x_r43 = F.relu(self.cnn_enc_43_ref(x_r42))

        x = inputs_tar

        x = F.relu(self.cnn_enc_11_tar(x))
        x = F.relu(self.cnn_enc_12_tar(x))

        x = torch.cat([x, x_r12], dim=1)
        if self.use_attn[0]:
            x = F.relu(self.attn_1(x)[0])

        x = F.relu(self.cnn_enc_21_tar(x))
        x = F.relu(self.cnn_enc_22_tar(x))

        x = torch.cat([x, x_r22], dim=1)
        if self.use_attn[1]:
            x = F.relu(self.attn_2(x)[0])

        x = F.relu(self.cnn_enc_31_tar(x))
        x = F.relu(self.cnn_enc_32_tar(x))
        x = F.relu(self.cnn_enc_33_tar(x))

        x = torch.cat([x, x_r33], dim=1)
        if self.use_attn[2]:
            x = F.relu(self.attn_3(x)[0])

        x = F.relu(self.cnn_enc_41_tar(x))
        x = F.relu(self.cnn_enc_42_tar(x))
        x = F.relu(self.cnn_enc_43_tar(x))

        x = torch.cat([x, x_r43], dim=1)
        if self.use_attn[3]:
            x = F.relu(self.attn_4(x)[0])

        x = F.relu(self.cnn_enc_51_tar(x))
        x = F.relu(self.cnn_enc_52_tar(x))
        x = F.relu(self.cnn_enc_53_tar(x))

        # x = x.view(-1, 512 * 4 * 4)
        x = torch.reshape(x, (-1, 512 * self.feature_size * self.feature_size))

        x = self.fc(x)

        return x


class CNN_Encoder(nn.Module):
    def __init__(self, input_dim, output_dim, input_size, first_kernel_size, first_padding, use_atrous):
        super(CNN_Encoder, self).__init__()

        if use_atrous:
            atrou_rates = [1, 1, 2, 4, 4]
        else:
            atrou_rates = [1, 1, 1, 1, 1]

        self.cnn_enc_11 = general_conv2d(input_dim, 32, kernel_size=first_kernel_size, stride=2, padding=first_padding, atrous=use_atrous, atrous_rate=atrou_rates[0])
        self.cnn_enc_12 = general_conv2d(32, 32, kernel_size=3, stride=1, atrous=use_atrous, atrous_rate=atrou_rates[0])

        self.cnn_enc_21 = general_conv2d(32, 64, kernel_size=3, stride=2, atrous=use_atrous, atrous_rate=atrou_rates[1])
        self.cnn_enc_22 = general_conv2d(64, 64, kernel_size=3, stride=1, atrous=use_atrous, atrous_rate=atrou_rates[1])

        self.cnn_enc_31 = general_conv2d(64, 128, kernel_size=3, stride=2, atrous=use_atrous, atrous_rate=atrou_rates[2])
        self.cnn_enc_32 = general_conv2d(128, 128, kernel_size=3, stride=1, atrous=use_atrous, atrous_rate=atrou_rates[2])
        self.cnn_enc_33 = general_conv2d(128, 128, kernel_size=3, stride=1, atrous=use_atrous, atrous_rate=atrou_rates[2])

        self.cnn_enc_41 = general_conv2d(128, 256, kernel_size=3, stride=2, atrous=use_atrous, atrous_rate=atrou_rates[3])
        self.cnn_enc_42 = general_conv2d(256, 256, kernel_size=3, stride=1, atrous=use_atrous, atrous_rate=atrou_rates[3])
        self.cnn_enc_43 = general_conv2d(256, 256, kernel_size=3, stride=1, atrous=use_atrous, atrous_rate=atrou_rates[3])

        self.cnn_enc_51 = general_conv2d(256, 512, kernel_size=3, stride=2, atrous=use_atrous, atrous_rate=atrou_rates[4])
        self.cnn_enc_52 = general_conv2d(512, 512, kernel_size=3, stride=1, atrous=use_atrous, atrous_rate=atrou_rates[4])
        self.cnn_enc_53 = general_conv2d(512, 512, kernel_size=3, stride=1, atrous=use_atrous, atrous_rate=atrou_rates[4])

        assert input_size % 32 == 0
        self.feature_size = input_size // 32
        self.fc = nn.Linear(512 * self.feature_size * self.feature_size, output_dim)

        self.use_attn = [False for _ in range(4)]

    def forward(self, inputs):
        x = inputs

        x = F.relu(self.cnn_enc_11(x))
        # print('cnn_enc_11', x.size())
        x = F.relu(self.cnn_enc_12(x))
        # print('cnn_enc_12', x.size())
        if self.use_attn[0]:
            x = F.relu(self.attn_1(x)[0])

        x = F.relu(self.cnn_enc_21(x))
        # print('cnn_enc_21', x.size())
        x = F.relu(self.cnn_enc_22(x))
        # print('cnn_enc_22', x.size())
        if self.use_attn[1]:
            x = F.relu(self.attn_2(x)[0])

        x = F.relu(self.cnn_enc_31(x))
        # print('cnn_enc_31', x.size())
        x = F.relu(self.cnn_enc_32(x))
        # print('cnn_enc_32', x.size())
        x = F.relu(self.cnn_enc_33(x))
        # print('cnn_enc_33', x.size())
        if self.use_attn[2]:
            x = F.relu(self.attn_3(x)[0])

        x = F.relu(self.cnn_enc_41(x))
        # print('cnn_enc_41', x.size())
        x = F.relu(self.cnn_enc_42(x))
        # print('cnn_enc_42', x.size())
        x = F.relu(self.cnn_enc_43(x))
        # print('cnn_enc_43', x.size())
        if self.use_attn[3]:
            x = F.relu(self.attn_4(x)[0])

        x = F.relu(self.cnn_enc_51(x))
        # print('cnn_enc_51', x.size())
        x = F.relu(self.cnn_enc_52(x))
        # print('cnn_enc_52', x.size())
        x = F.relu(self.cnn_enc_53(x))
        # print('cnn_enc_53', x.size())

        # x = x.view(-1, 512 * 4 * 4)
        x = torch.reshape(x, (-1, 512 * self.feature_size * self.feature_size))
        # print('x', x.size())

        x = self.fc(x)
        # print('fc', x.size())

        return x


class RNN_Decoder(nn.Module):
    def __init__(self, input_size, dec_rnn_size, output_size, is_hyper=False, zero_init='none'):
        super(RNN_Decoder, self).__init__()
        self.input_size = input_size
        self.dec_rnn_size = dec_rnn_size
        self.is_hyper = is_hyper

        if not is_hyper:
            self.lstm = nn.LSTMCell(input_size, dec_rnn_size)
        else:
            self.lstm = HyperLSTMCell(input_size, dec_rnn_size)

        self.dec_fc_params = nn.Linear(dec_rnn_size, output_size)

        if zero_init == 'final':
            for (m_name, m) in self.named_modules():
                if isinstance(m, nn.Linear) and m_name == 'dec_fc_params':
                    nn.init.constant_(m.weight, 0)
                    nn.init.constant_(m.bias, 0)
        elif zero_init == 'all':
            for (m_name, m) in self.named_modules():
                if isinstance(m, nn.Linear):
                    nn.init.constant_(m.weight, 0)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.LayerNorm):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
        else:
            assert zero_init == 'none'
        # print('RNN_Decoder zero init:', zero_init)

    def forward(self, input_x, in_state=None):
        """
        :param input_x: (batch_size, input_size)
        :param in_state: (h0, c0) / (h0, c0, h0_hat, c0_hat)
        :return:
        """
        input_state = in_state
        if not self.is_hyper:
            rnn_hidden, cell_state = self.lstm(input_x, input_state)  # (N, dec_rnn_size)
            output_state = (rnn_hidden, cell_state)
        else:
            input_state_h, input_state_h_hat, input_state_c, input_state_c_hat = input_state
            rnn_hidden, cell_state, rnn_hidden_hat, cell_state_hat = self.lstm(
                input_x, input_state_h, input_state_c, input_state_h_hat, input_state_c_hat)  # each with (N, dec_rnn_size)
            output_state = (rnn_hidden, rnn_hidden_hat, cell_state, cell_state_hat)

        output = self.dec_fc_params(rnn_hidden)  # (N, n_out)
        return output, output_state


class MLP_Decoder(nn.Module):
    def __init__(self, input_size, output_size, zero_init='none'):
        super(MLP_Decoder, self).__init__()
        self.input_size = input_size

        hidden_size = 128
        self.dec_fc_1 = nn.Linear(input_size, hidden_size)
        # self.dec_fc_2 = nn.Linear(hidden_size, hidden_size)
        self.dec_fc_params = nn.Linear(hidden_size, output_size)

        if zero_init == 'last':
            for (m_name, m) in self.named_modules():
                if isinstance(m, nn.Linear) and m_name == 'dec_fc_params':
                    nn.init.constant_(m.weight, 0)
                    nn.init.constant_(m.bias, 0)
        elif zero_init == 'all':
            for (m_name, m) in self.named_modules():
                if isinstance(m, nn.Linear):
                    nn.init.constant_(m.weight, 0)
                    nn.init.constant_(m.bias, 0)
        else:
            assert zero_init == 'none'
        # print('MLP_Decoder zero init:', zero_init)

    def forward(self, input_x):
        """
        :param input_x: (batch_size, input_size)
        :return:
        """
        features_1 = self.dec_fc_1(input_x)
        # features_2 = self.dec_fc_2(features_1)
        output = self.dec_fc_params(features_1)  # (N, n_out)
        return output


class Correspondence_Model(nn.Module):
    def __init__(self, hps):
        super(Correspondence_Model, self).__init__()
        self.hps = hps

        transform_out_size = 1 if self.hps.use_square_window else 2  # scaling
        transform_out_size += 2  # translation
        if self.hps.transform_with_rotation:
            transform_out_size += 1

        first_kernel_size = self.hps.first_kernel_size
        first_padding = (first_kernel_size - 1) // 2

        # transform encoder
        if self.hps.enc_model_transform == 'combined':
            cnn_in_size = 2
            if self.hps.add_coordconv:
                cnn_in_size += 2
            cnn_out_size = self.hps.z_size

            self.encoder_transform = CNN_Encoder(cnn_in_size, cnn_out_size, input_size=self.hps.raster_size_corres,
                                                 first_kernel_size=first_kernel_size, first_padding=first_padding,
                                                 use_atrous=self.hps.use_atrous_conv)
        else:
            raise Exception('Unknown enc_model_transform:', self.hps.enc_model_transform)

        dec_in_size = self.hps.z_size

        if self.hps.dec_model_transform == 'mlp':
            self.decoder_transform = MLP_Decoder(dec_in_size, transform_out_size,
                                                 zero_init=self.hps.transform_module_zero_init)
        else:
            raise Exception('Unknown dec_model_transform:', self.hps.dec_model_transform)

        if self.hps.enc_model_correspondence == 'separated':
            cnn_in_size_ref = 2
            cnn_in_size_tar = 1
            if self.hps.use_segment_img:
                cnn_in_size_ref += 1
            if self.hps.use_reference_canvas:
                cnn_in_size_ref += 1
            if self.hps.use_target_canvas:
                cnn_in_size_tar += 1
            if self.hps.add_coordconv:
                cnn_in_size_ref += 2
                cnn_in_size_tar += 2
            cnn_out_size = 2

            self.encoder = CNN_SepEncoder_correspondence(cnn_in_size_ref, cnn_in_size_tar, cnn_out_size, input_size=self.hps.raster_size_corres,
                                                         use_atrous=self.hps.use_atrous_conv,
                                                         use_attn=self.hps.use_attn_corres, attn_type=self.hps.attn_type_corres,
                                                         sa_block_pos=self.hps.sa_block_pos_corres,
                                                         use_dropout=self.hps.use_dropout, dropout_rate=self.hps.dropout_rate)
        else:
            raise Exception('Unknown enc_model_correspondence:', self.hps.enc_model_correspondence)

        if self.hps.add_coordconv:
            self.coordconv_input = self.get_coordconv()  # (2, image_size, image_size)

    def forward(self, reference_images, reference_dot_images, reference_segment_images, reference_canvas_images,
                target_images, target_canvas_images,
                cursor_position_ref, image_size, init_trans_window_sizes):
        """
        :param reference_images: (N, H, W, 1), float32, [0.0-stroke, 1.0-BG]
        :param reference_dot_images: (N, H_c, W_c, 1), float32, [0.0-stroke, 1.0-BG]
        :param reference_segment_images: (N, H, W, 1), float32, [0.0-stroke, 1.0-BG]
        :param reference_canvas_images: (N, H, W, 1), float32, [0.0-stroke, 1.0-BG]
        :param target_images: (N, H, W, 1), float32, [0.0-stroke, 1.0-BG]
        :param target_canvas_images: (N, H, W, 1), [0.0-stroke, 1.0-BG]
        :param cursor_position_ref: (N, 1, 2), in size [0.0, 1.0]
        :param init_trans_window_sizes: (1, 1, 2)
        """
        # ================== Stage-1: Transformation ================== #
        # reference_images, target_images: (N, H, W, 1), [0.0-stroke, 1.0-BG]
        crop_inputs_trans = torch.cat([reference_images, target_images], dim=-1)  # (N, H, W, *)
        cropped_outputs_trans = image_cropping_stn(cursor_position_ref, crop_inputs_trans, image_size,
                                                   init_trans_window_sizes, raster_size=self.hps.raster_size_corres)
        curr_patch_input_ref_trans = cropped_outputs_trans[:, :, :, 0:1]  # (N, raster_size, raster_size, 1), [0.0-stroke, 1.0-BG]
        curr_patch_input_tar_trans = cropped_outputs_trans[:, :, :, 1:2]  # (N, raster_size, raster_size, 1), [0.0-stroke, 1.0-BG]
        curr_patch_input_ref_trans_in = normalize_image_m1to1(curr_patch_input_ref_trans)
        curr_patch_input_tar_trans_in = normalize_image_m1to1(curr_patch_input_tar_trans)
        # (N, raster_size, raster_size, 1), [-1.0-stroke, 1.0-BG]

        ## generate the transformation of target window size
        transform_z = self.build_encoder_transform(curr_patch_input_ref_trans_in, curr_patch_input_tar_trans_in)
        transform_output, _ = self.build_decoder_transform(transform_z, None)
        # transform_output: (N, 5)

        transform_output_translation = transform_output[:, 0:2]  # (N, 2)
        transform_output_scaling = transform_output[:, 2:4]  # (N, 2)
        if self.hps.transform_with_rotation:
            transform_output_rotate_angle = transform_output[:, 4:5]  # (N, 1)

        ## Then, use a small window to crop patches for the correspondence
        corres_window_sizes_ref = torch.tensor([self.hps.raster_size_corres, self.hps.raster_size_corres]).float()
        corres_window_sizes_ref = corres_window_sizes_ref.unsqueeze(dim=0).unsqueeze(dim=0).cuda()  # (1, 1, 2)

        ## Reference
        crop_inputs_corres_ref = torch.cat([reference_images, reference_segment_images, reference_canvas_images], dim=-1)  # (N, H, W, *)
        crop_outputs_corres_ref = image_cropping_stn(cursor_position_ref, crop_inputs_corres_ref, image_size,
                                                     corres_window_sizes_ref, raster_size=self.hps.raster_size_corres)
        reference_images_patch_corres = crop_outputs_corres_ref[:, :, :, 0:1]
        reference_segment_images_patch_corres = crop_outputs_corres_ref[:, :, :, 1:2]
        reference_canvas_images_patch_corres = crop_outputs_corres_ref[:, :, :, 2:3]
        # reference_images_patch_corres: (N, H_c, W_c, 1), [0-stroke, 1-BG]
        # reference_segment_images_patch_corres: (N, H_c, W_c, 1), [0-stroke, 1-BG]
        # reference_canvas_images_patch_corres: (N, H_c, W_c, 1), [0-stroke, 1-BG]

        ## Target
        # Translation
        pred_window_translate = torch.tanh(transform_output_translation)  # (N, 2), [-1.0, 1.0]
        pred_window_translate = pred_window_translate.unsqueeze(dim=1) * (init_trans_window_sizes / 2.0)  # (N, 1, 2), in full size
        pred_cursor_position_tar = cursor_position_ref * image_size + pred_window_translate  # (N, 1, 2), in full size
        # print(' >> Correspondence | pred_cursor_position_tar', pred_cursor_position_tar)
        pred_cursor_position_tar = pred_cursor_position_tar / float(image_size)  # (N, 1, 2), [0.0, 1.0]

        # Scaling
        pred_window_scaling_times_tar = torch.tanh(transform_output_scaling)  # (N, 2), [-1.0, 1.0]
        pred_window_scaling_times_tar = (pred_window_scaling_times_tar + 1.0) / 2.0 * self.hps.window_size_scaling_times_tar[1]  # (N, 2), [0.0, 2.0]
        pred_window_scaling_times_tar = torch.clamp(pred_window_scaling_times_tar, self.hps.window_size_scaling_times_tar[0], self.hps.window_size_scaling_times_tar[1])  # (N, 2), [0.2, 2.0]
        # print(' >> Correspondence | pred_window_scaling_times_tar', pred_window_scaling_times_tar)

        curr_window_size_tar_pred = pred_window_scaling_times_tar.unsqueeze(dim=1) * corres_window_sizes_ref  # (N, 1, 2), in full size
        curr_window_size_tar_pred = torch.max(curr_window_size_tar_pred, torch.tensor(self.hps.window_size_min).float().cuda())
        curr_window_size_tar_pred = torch.min(curr_window_size_tar_pred, torch.tensor(image_size * 2.0).float().cuda())

        # Rotation
        if self.hps.transform_with_rotation:
            pred_window_rotate_angle_tar = torch.tanh(transform_output_rotate_angle)  # (N, 1), [-1.0, 1.0]
            pred_window_rotate_angle_tar = torch.mul(pred_window_rotate_angle_tar, 180.0)  # (N, 1), [-180.0, 180.0]
            # print(' >> Correspondence | pred_window_rotate_angle_tar', pred_window_rotate_angle_tar)
        else:
            pred_window_rotate_angle_tar = None

        ## crop the target again
        crop_inputs_corres_tar = torch.cat([target_images, target_canvas_images], dim=-1)  # (N, H, W, *)
        cropped_outputs_corres_tar = image_cropping_stn(pred_cursor_position_tar, crop_inputs_corres_tar, image_size,
                                                        curr_window_size_tar_pred, raster_size=self.hps.raster_size_corres,
                                                        rotation_angle=pred_window_rotate_angle_tar)
        target_images_patch_corres = cropped_outputs_corres_tar[:, :, :, 0:1]
        target_canvas_images_patch_corres = cropped_outputs_corres_tar[:, :, :, 1:2]
        # target_images_patch_corres: (N, H_c, W_c, 1), [0-stroke, 1-BG]
        # target_canvas_images_patch_corres: (N, H_c, W_c, 1), [0-stroke, 1-BG]

        # ================== Stage-2: Correspondence ================== #
        reference_images_patch_corres_in = normalize_image_m1to1(reference_images_patch_corres)
        reference_dot_img_patch_corres_in = normalize_image_m1to1(reference_dot_images)
        reference_segment_images_patch_corres_in = normalize_image_m1to1(reference_segment_images_patch_corres)
        reference_canvas_images_patch_corres_in = normalize_image_m1to1(reference_canvas_images_patch_corres)
        target_images_patch_corres_in = normalize_image_m1to1(target_images_patch_corres)
        target_canvas_images_patch_corres_in = normalize_image_m1to1(target_canvas_images_patch_corres)
        # (N, H, W, 1), [-1.0-stroke, 1.0-BG]

        if self.hps.enc_model_correspondence == 'separated':
            batch_input_ref_list = [reference_images_patch_corres_in]
            if self.hps.use_reference_canvas:
                batch_input_ref_list.append(reference_canvas_images_patch_corres_in)
            if self.hps.use_segment_img:
                batch_input_ref_list.append(reference_segment_images_patch_corres_in)
            batch_input_ref_list.append(reference_dot_img_patch_corres_in)
            batch_input_ref = torch.cat(batch_input_ref_list, dim=-1)  # (N, H, W, *), [-1.0-stroke, 1.0-BG]

            if self.hps.use_target_canvas:
                batch_input_tar = torch.cat([target_images_patch_corres_in, target_canvas_images_patch_corres_in], dim=-1)  # (N, H, W, 2), [-1.0-stroke, 1.0-BG]
            else:
                batch_input_tar = target_images_patch_corres_in  # (N, H, W, 1), [-1.0-stroke, 1.0-BG]

            # transform to nchw
            batch_input_ref = batch_input_ref.permute(0, 3, 1, 2)  # (N, *, H, W), [-1.0-stroke, 1.0-BG]
            batch_input_tar = batch_input_tar.permute(0, 3, 1, 2)  # (N, *, H, W), [-1.0-stroke, 1.0-BG]
            if self.hps.add_coordconv:
                batch_input_ref = self.add_coords(batch_input_ref)  # (N, in_dim + 2, in_H, in_W)
                batch_input_tar = self.add_coords(batch_input_tar)  # (N, in_dim + 2, in_H, in_W)
            pred_params_trans = self.encoder(batch_input_ref, batch_input_tar)  # (N, 2), [-1.0, 1.0]
        else:
            raise Exception('Unknown enc_model_correspondence:', self.hps.enc_model_correspondence)

        if self.hps.use_clinging:
            pred_params_trans_np = pred_params_trans.cpu().data.numpy()
            target_images_patch_corres_np = target_images_patch_corres.cpu().data.numpy()
            pred_params_trans_np = correspondence_clinging(target_images_patch_corres_np, pred_params_trans_np,
                                                           self.hps.raster_size_corres,
                                                           binary_threshold=self.hps.clinging_binary_threshold)
            pred_params_trans = torch.tensor(pred_params_trans_np).float().cuda()  # (N, 2), [-1.0, 1.0]

        ## Reversed Transformation
        if self.hps.transform_with_rotation:
            pred_params_rel = spatial_transform_reverse_point(pred_params_trans, pred_window_rotate_angle_tar)  # (N, 2), [-1.0+, 1.0+]
        else:
            pred_params_rel = pred_params_trans
        pred_params_offset_global = pred_params_rel * (curr_window_size_tar_pred.squeeze(dim=1) / 2.0)  # (N, 2)
        pred_params_global = pred_cursor_position_tar.squeeze(dim=1) * float(image_size) + pred_params_offset_global  # (N, 2), in full size
        pred_params_global = pred_params_global / float(image_size)  # (N, 2), in [0.0, 1.0]

        return pred_params_global

    def get_coordconv(self):
        xx_ones = torch.ones(self.hps.raster_size_corres, dtype=torch.int32)  # e.g. (image_size)
        xx_ones = xx_ones.unsqueeze(dim=-1)  # e.g. (image_size, 1)
        xx_range = torch.arange(self.hps.raster_size_corres, dtype=torch.int32)  # e.g. (image_size)
        xx_range = xx_range.unsqueeze(0)  # e.g. (1, image_size)

        xx_channel = torch.matmul(xx_ones, xx_range)  # e.g. (image_size, image_size)
        xx_channel = xx_channel.unsqueeze(0)  # e.g. (1, image_size, image_size)

        yy_ones = torch.ones(self.hps.raster_size_corres, dtype=torch.int32)  # e.g. (image_size)
        yy_ones = yy_ones.unsqueeze(0)  # e.g. (1, image_size)
        yy_range = torch.arange(self.hps.raster_size_corres, dtype=torch.int32)  # (image_size)
        yy_range = yy_range.unsqueeze(-1)  # e.g. (image_size, 1)

        yy_channel = torch.matmul(yy_range, yy_ones)  # e.g. (image_size, image_size)
        yy_channel = yy_channel.unsqueeze(0)  # e.g. (1, image_size, image_size)

        xx_channel = xx_channel.float() / (self.hps.raster_size_corres - 1)
        yy_channel = yy_channel.float() / (self.hps.raster_size_corres - 1)
        xx_channel = xx_channel * 2 - 1  # [-1, 1]
        yy_channel = yy_channel * 2 - 1

        # xx_channel = xx_channel.cuda()
        # yy_channel = yy_channel.cuda()

        ret = torch.cat([
            xx_channel,
            yy_channel,
        ], dim=0)  # (2, image_size, image_size)
        ret = ret.detach()

        return ret

    def add_coords(self, input_tensor):
        batch_size = input_tensor.size()[0]  # get N size
        coords = torch.unsqueeze(self.coordconv_input, dim=0).repeat(batch_size, 1, 1, 1)  # (N, 2, image_size, image_size)
        coords = coords.to(input_tensor.device)
        result = torch.cat([input_tensor, coords], dim=1)  # (N, C+2, image_size, image_size)
        return result

    def build_encoder_transform(self, patch_input_ref, patch_input_tar):
        """
        :param patch_input_ref & patch_input_tar: (N, raster_size, raster_size, 1), [-1.0-stroke, 1.0-BG]
        :return:
        """
        # transform to nchw
        patch_inputs_ref = patch_input_ref  # (N, raster_size, raster_size, 1), [-1.0-stroke, 1.0-BG]
        patch_inputs_ref = patch_inputs_ref.permute(0, 3, 1, 2)  # (N, 1, raster_size, raster_size), [-1.0-stroke, 1.0-BG]
        patch_inputs_tar = patch_input_tar  # (N, raster_size, raster_size, 1), [-1.0-stroke, 1.0-BG]
        patch_inputs_tar = patch_inputs_tar.permute(0, 3, 1, 2)  # (N, 1, raster_size, raster_size), [-1.0-stroke, 1.0-BG]

        if self.hps.enc_model_transform == 'combined':
            batch_input = torch.cat([patch_inputs_ref, patch_inputs_tar], dim=1)  # (N, 4, raster_size, raster_size), [-1.0-stroke, 1.0-BG]

            if self.hps.add_coordconv:
                batch_input = self.add_coords(batch_input)  # (N, in_dim + 2, in_H, in_W)
            output = self.encoder_transform(batch_input)  # (N, z_size)
        else:
            raise Exception('Unknown enc_model_transform:', self.hps.enc_model_transform)

        return output

    def build_decoder_transform(self, dec_input, prev_state):
        """
        :param dec_input: (N, in_dim)
        :return:
        """
        h_output = self.decoder_transform(dec_input)
        next_state = None
        return h_output, next_state


class Generative_Model(nn.Module):
    def __init__(self, hps, corres_module):
        super(Generative_Model, self).__init__()
        self.hps = hps
        self.stroke_thickness = hps.stroke_thickness
        self.correspondence_module = corres_module

        if self.hps.data_type in ['TU-Derlin', 'TU-Refined']:
            self.color_rgb_set = generate_colors2(40)

        transform_out_size = 1 if self.hps.use_square_window else 2
        if self.hps.transform_with_rotation:
            transform_out_size += 1

        first_kernel_size = self.hps.first_kernel_size
        first_padding = (first_kernel_size - 1) // 2

        # transform encoder
        if self.hps.enc_model_transform == 'combined':
            cnn_in_size = 2
            if self.hps.transform_use_global_info:
                cnn_in_size += 1
            if self.hps.add_coordconv:
                cnn_in_size += 2
            cnn_out_size = self.hps.z_size

            self.encoder_transform = CNN_Encoder(cnn_in_size, cnn_out_size, input_size=self.hps.raster_size,
                                                 first_kernel_size=first_kernel_size, first_padding=first_padding,
                                                 use_atrous=self.hps.use_atrous_conv)
        else:
            raise Exception('Unknown enc_model_transform:', self.hps.enc_model_transform)

        # tracing encoder
        if self.hps.enc_model_tracing == 'separated':
            cnn_in_size_ref = 3
            cnn_in_size_tar_end = 2
            cnn_in_size_tar_ctrl = 3
            if self.hps.add_coordconv:
                cnn_in_size_ref += 2
                cnn_in_size_tar_end += 2
                cnn_in_size_tar_ctrl += 2
            cnn_out_size = self.hps.z_size

            self.encoder_end = CNN_SepEncoder(cnn_in_size_ref, cnn_in_size_tar_end, cnn_out_size, input_size=self.hps.raster_size,
                                              first_kernel_size=first_kernel_size, first_padding=first_padding,
                                              use_atrous=self.hps.use_atrous_conv)
            self.encoder_ctrl = CNN_SepEncoder(cnn_in_size_ref, cnn_in_size_tar_ctrl, cnn_out_size, input_size=self.hps.raster_size,
                                               first_kernel_size=first_kernel_size, first_padding=first_padding,
                                               use_atrous=self.hps.use_atrous_conv)
        else:
            raise Exception('Unknown enc_model_tracing:', self.hps.enc_model_tracing)

        if self.hps.add_coordconv:
            self.coordconv_input = self.get_coordconv()  # (2, raster_size, raster_size)

        dec_in_size = self.hps.z_size
        dec_out_size_end = 2
        dec_out_size_ctrl = 4

        is_hyper = True if self.hps.rnn_model == 'hyper' else False
        if self.hps.dec_model_transform == 'mlp':
            self.decoder_transform = MLP_Decoder(dec_in_size, transform_out_size, zero_init=self.hps.transform_module_zero_init)
        else:
            raise Exception('Unknown dec_model_transform:', self.hps.dec_model_transform)

        if self.hps.dec_model_tracing == 'rnn':
            self.decoder_end = RNN_Decoder(dec_in_size, self.hps.dec_rnn_size, dec_out_size_end, is_hyper=is_hyper)
            self.decoder_ctrl = RNN_Decoder(dec_in_size, self.hps.dec_rnn_size, dec_out_size_ctrl, is_hyper=is_hyper)
        else:
            raise Exception('Unknown dec_model_tracing:', self.hps.dec_model_tracing)

    def forward(self, seq_num, reference_images, target_images,
                reference_dot_images_patch, reference_segment_images,
                endpoints_pos_ref, starting_states, base_window_size, model_mode, image_size):
        """
        :param reference_images: (N, H, W, 1), float32, [0.0-stroke, 1.0-BG]
        :param target_images: (N, H, W, 1), float32, [0.0-stroke, 1.0-BG]
        :param reference_segment_images: (N, seq_num, H, W), [0.0-stroke, 1.0-BG]
        :param reference_dot_images_patch: (N, H_c, W_c), [0.0-stroke, 1.0-BG]
        :param endpoints_pos_ref: (N, seq_num, 2), float32, in [0.0, 1.0]
        :param starting_states: (N, seq_num), {1.0, 0.0}
        :param base_window_size: (N, seq_num), float32, in [0.0, 1.0]
        :return:
        """
        self.model_mode = model_mode
        assert model_mode in ['eval', 'inference']

        if self.hps.data_type not in ['TU-Derlin', 'TU-Refined']:
            self.color_rgb_set = generate_colors2(seq_num)  # (seq_num, 3), in [0., 1.]

        pred_params, pred_raster_images, pred_raster_images_rgb = \
            self.get_points_and_raster_image(seq_num, reference_images, target_images, reference_segment_images,
                                             reference_dot_images_patch, endpoints_pos_ref,
                                             starting_states, base_window_size, image_size)
        # pred_params: (N, seq_num, 4, 2), in full size
        # pred_raster_images: (N, H, W), [0.0-BG, 1.0-stroke]
        # pred_raster_images_rgb: (N, H, W, 3), [0.0-BG, 1.0-stroke]

        pred_raster_images = 1.0 - pred_raster_images  # (N, H, W), [0.0-stroke, 1.0-BG]
        pred_raster_images_rgb = 1.0 - pred_raster_images_rgb  # (N, H, W, 3), [0.0-stroke, 1.0-BG]

        return pred_raster_images, pred_raster_images_rgb, pred_params

    def get_points_and_raster_image(self, seq_num,
                                    reference_images, target_images, reference_segment_images,
                                    reference_dot_images_patch, endpoints_pos_ref,
                                    starting_states, base_window_size, image_size):
        """
        :param reference_images: (N, H, W, 1), float32, [0.0-stroke, 1.0-BG]
        :param target_images: (N, H, W, 1), float32, [0.0-stroke, 1.0-BG]
        :param reference_segment_images: (N, seq_num, H, W), [0.0-stroke, 1.0-BG]
        :param reference_dot_images_patch: (N, H_c, W_c), [0.0-stroke, 1.0-BG]
        :param endpoints_pos_ref: (N, seq_num, 2), float32, in [0.0, 1.0]
        :param starting_states: (N, seq_num), {1.0, 0.0}
        :param base_window_size: (N, seq_num), float32, in [0.0, 1.0]
        :return:
        """
        zero_state = torch.zeros(reference_images.size(0), self.hps.dec_rnn_size).cuda()

        if not self.hps.rnn_model == 'hyper':
            next_state_end = (zero_state, zero_state)
            next_state_ctrl = (zero_state, zero_state)
            transform_next_state = (zero_state, zero_state)
        else:
            next_state_end = (zero_state, zero_state, zero_state, zero_state)
            next_state_ctrl = (zero_state, zero_state, zero_state, zero_state)
            transform_next_state = (zero_state, zero_state, zero_state, zero_state)

        corres_window_size = get_correspondence_window_size(image_size, self.hps.init_window_size_corres_trans)
        corres_window_sizes = torch.tensor([corres_window_size, corres_window_size]).float()
        corres_window_sizes = corres_window_sizes.unsqueeze(dim=0).unsqueeze(dim=0).cuda()  # (1, 1, 2)

        segment_params_list = []

        curr_canvas_ref = torch.squeeze(torch.zeros_like(reference_images), dim=-1)  # (N, H, W), [0.0-BG, 1.0-stroke]

        curr_canvas_tar_black = torch.squeeze(torch.zeros_like(target_images), dim=-1)  # (N, H, W), [0.0-BG, 1.0-stroke]
        curr_canvas_tar_rgb = torch.zeros_like(target_images)
        curr_canvas_tar_rgb = curr_canvas_tar_rgb.repeat(1, 1, 1, 3)  # (N, H, W, 3), [0.0-BG, 1.0-stroke]

        cursor_position_loop_tar_inference = None
        for seq_i in tqdm(range(seq_num)):
            # reference cursor position
            cursor_position_loop_ref = endpoints_pos_ref[:, seq_i:seq_i + 1, :]  # (N, 1, 2), in size [0.0, 1.0]

            # reference segment image
            curr_segment_image_ref = reference_segment_images[:, seq_i, :, :]  # (N, H, W), [0.0-stroke, 1.0-BG]
            curr_segment_image_ref = curr_segment_image_ref.unsqueeze(dim=-1)  # (N, H, W, 1), [0.0-stroke, 1.0-BG]

            # canvas images
            curr_canvas_ref_for_crop = 1.0 - curr_canvas_ref.unsqueeze(dim=-1)  # (N, H, W, 1), [0.0-stroke, 1.0-BG]
            curr_canvas_tar_for_crop = 1.0 - curr_canvas_tar_black.unsqueeze(dim=-1)  # (N, H, W, 1), [0.0-stroke, 1.0-BG]

            # ================== Stage-1: Starting point correspondence ================== #
            if starting_states[0, seq_i] == 1:
                # print('Starting point correspondence:', seq_i)
                reference_dot_images_patch_corres = torch.unsqueeze(reference_dot_images_patch, dim=-1)  # (N, H_c, W_c, 1), [0-stroke, 1-BG]
                pred_params_corres = self.correspondence_module(reference_images=reference_images,
                                                                reference_dot_images=reference_dot_images_patch_corres,
                                                                reference_segment_images=curr_segment_image_ref,
                                                                reference_canvas_images=curr_canvas_ref_for_crop,
                                                                target_images=target_images,
                                                                target_canvas_images=curr_canvas_tar_for_crop,
                                                                cursor_position_ref=cursor_position_loop_ref,
                                                                image_size=image_size,
                                                                init_trans_window_sizes=corres_window_sizes)
                # pred_params_corres: (N, 2), in [0.0, 1.0]
                cursor_position_loop_tar_inference = pred_params_corres.unsqueeze(dim=1)  # (N, 1, 2), in [0.0, 1.0]

            # =================== Stage-2: Transformation and Tracing =================== #
            ## Reference processing
            curr_window_size_ref_base = base_window_size[:, seq_i:seq_i + 1].unsqueeze(dim=-1)  # (N, 1, 1), in [0.0, 1.0]
            curr_window_size_ref_base = torch.mul(curr_window_size_ref_base, image_size)  # (N, 1, 1), in full size
            curr_window_size_ref = torch.mul(curr_window_size_ref_base, self.hps.window_size_scaling_ref)  # (N, 1, 1), in full size
            curr_window_size_ref = torch.max(curr_window_size_ref, torch.tensor(self.hps.window_size_min).float().cuda())
            curr_window_size_ref = torch.min(curr_window_size_ref, torch.tensor(image_size * 2.0).float().cuda())
            curr_window_size_ref = torch.cat([curr_window_size_ref, curr_window_size_ref], dim=-1)  # (N, 1, 2), in full size

            # reference_images: (N, H, W, 1), [0.0-stroke, 1.0-BG]
            crop_inputs_ref_raw = torch.cat([reference_images, curr_segment_image_ref, curr_canvas_ref_for_crop], dim=-1)  # (N, H, W, *)
            crop_inputs_ref = crop_inputs_ref_raw
            cropped_outputs_ref = image_cropping_stn(cursor_position_loop_ref, crop_inputs_ref, image_size, curr_window_size_ref,
                                                     raster_size=self.hps.raster_size)
            curr_patch_input_ref = cropped_outputs_ref[:, :, :, 0:1]  # (N, raster_size, raster_size, 1), [0.0-stroke, 1.0-BG]
            curr_patch_segment_ref = cropped_outputs_ref[:, :, :, 1:2]  # (N, raster_size, raster_size, 1), [0.0-stroke, 1.0-BG]
            curr_patch_canvas_ref = cropped_outputs_ref[:, :, :, 2:3]  # (N, raster_size, raster_size, 1), [0.0-stroke, 1.0-BG]

            curr_patch_input_ref = normalize_image_m1to1(curr_patch_input_ref)
            curr_patch_segment_ref = normalize_image_m1to1(curr_patch_segment_ref)
            curr_patch_canvas_ref = normalize_image_m1to1(curr_patch_canvas_ref)
            # (N, raster_size, raster_size, 1), [-1.0-stroke, 1.0-BG]

            # Update the reference canvas
            curr_canvas_ref = torch.add(curr_canvas_ref, 1.0 - curr_segment_image_ref.squeeze(dim=-1))  # (N, H, W), [0.0-BG, 1.0-stroke]
            curr_canvas_ref = torch.clamp(curr_canvas_ref, 0.0, 1.0)  # (N, H, W), [0.0-BG, 1.0-stroke]

            ## Target processing
            # the given starting position for a new stroke; using loop for the original stroke
            starting_state = starting_states[:, seq_i:seq_i + 1].unsqueeze(dim=-1)  # (N, 1, 1), {1.0, 0.0}

            cursor_position_loop_tar = cursor_position_loop_tar_inference.detach()  # (N, 1, 2), in size [0.0, 1.0]

            curr_window_size_tar = torch.mul(curr_window_size_ref_base, self.hps.window_size_scaling_init_tar)  # (N, 1, 1), in full size
            curr_window_size_tar = torch.max(curr_window_size_tar, torch.tensor(self.hps.window_size_min).float().cuda())
            curr_window_size_tar = torch.min(curr_window_size_tar, torch.tensor(image_size * 2.0).float().cuda())
            curr_window_size_tar = torch.cat([curr_window_size_tar, curr_window_size_tar], dim=-1)  # (N, 1, 2), in full size

            curr_canvas_tar = 1.0 - curr_canvas_tar_black.detach()  # (N, H, W), [0.0-stroke, 1.0-BG]
            curr_canvas_tar = curr_canvas_tar.unsqueeze(dim=-1)  # (N, H, W, 1), [0.0-stroke, 1.0-BG]

            # target_images: (N, H, W, 1), [0.0-stroke, 1.0-BG]
            crop_inputs_tar_raw = torch.cat([target_images, curr_canvas_tar], dim=-1)  # (N, H, W, *)
            crop_inputs_tar_temp = crop_inputs_tar_raw
            cropped_outputs_tar_temp = image_cropping_stn(cursor_position_loop_tar, crop_inputs_tar_temp, image_size, curr_window_size_tar,
                                                          raster_size=self.hps.raster_size)
            curr_patch_input_tar_temp = cropped_outputs_tar_temp[:, :, :, 0:1]  # (N, raster_size, raster_size, 1), [0.0-stroke, 1.0-BG]
            curr_patch_canvas_tar_temp = cropped_outputs_tar_temp[:, :, :, 1:2]  # (N, raster_size, raster_size, 1), [0.0-stroke, 1.0-BG]

            curr_patch_input_tar_temp = normalize_image_m1to1(curr_patch_input_tar_temp)
            curr_patch_canvas_tar_temp = normalize_image_m1to1(curr_patch_canvas_tar_temp)
            # (N, raster_size, raster_size, 1), [-1.0-stroke, 1.0-BG]

            if self.hps.transform_use_global_info:
                curr_window_size_tar_global = curr_window_size_tar * self.hps.window_size_scaling_times_tar[1]  # (N, 1, 2), in full size
                curr_window_size_tar_global = torch.max(curr_window_size_tar_global, torch.tensor(self.hps.window_size_min).float().cuda())
                curr_window_size_tar_global = torch.min(curr_window_size_tar_global, torch.tensor(image_size * 2.0).float().cuda())

                cropped_outputs_tar_global = image_cropping_stn(cursor_position_loop_tar, crop_inputs_tar_temp, image_size, curr_window_size_tar_global,
                                                                raster_size=self.hps.raster_size)
                curr_patch_input_tar_global = cropped_outputs_tar_global[:, :, :, 0:1]  # (N, raster_size, raster_size, 1), [0.0-stroke, 1.0-BG]
                curr_patch_input_tar_global = normalize_image_m1to1(curr_patch_input_tar_global)
                # (N, raster_size, raster_size, 1), [-1.0-stroke, 1.0-BG]
            else:
                curr_patch_input_tar_global = None

            ## generate the transformation of target window size
            transform_prev_state = transform_next_state

            transform_z = self.build_encoder_transform(curr_patch_input_ref, curr_patch_input_tar_temp, curr_patch_input_tar_global)
            transform_output, transform_next_state = self.build_decoder_transform(transform_z, transform_prev_state)
            # transform_output: (N, 3)

            if self.hps.frozen_transform_module:
                transform_output = transform_output.detach()
            transform_output_scaling = transform_output[:, 0:2]  # (N, 2)
            if self.hps.transform_with_rotation:
                transform_output_rotate_angle = transform_output[:, 2:3]  # (N, 1)

            pred_window_scaling_times_tar = torch.tanh(transform_output_scaling)  # (N, 2), [-1.0, 1.0]
            pred_window_scaling_times_tar = (pred_window_scaling_times_tar + 1.0) / 2.0 * self.hps.window_size_scaling_times_tar[1]  # (N, 2), [0.0, 2.0]
            pred_window_scaling_times_tar = torch.clamp(pred_window_scaling_times_tar,
                                                        self.hps.window_size_scaling_times_tar[0], self.hps.window_size_scaling_times_tar[1])  # (N, 2), [0.2, 2.0]
            # print(' >> Tracing | pred_window_scaling_times_tar', pred_window_scaling_times_tar)

            curr_window_size_tar_pred = pred_window_scaling_times_tar.unsqueeze(dim=1) * curr_window_size_tar  # (N, 1, 2), in full size
            curr_window_size_tar_pred = torch.max(curr_window_size_tar_pred, torch.tensor(self.hps.window_size_min).float().cuda())
            curr_window_size_tar_pred = torch.min(curr_window_size_tar_pred, torch.tensor(image_size * 2.0).float().cuda())

            if self.hps.transform_with_rotation:
                pred_window_rotate_angle_tar = torch.tanh(transform_output_rotate_angle)  # (N, 1), [-1.0, 1.0]
                pred_window_rotate_angle_tar = torch.mul(pred_window_rotate_angle_tar, 180.0)  # (N, 1), [-180.0, 180.0]
                # print(' >> Tracing | pred_window_rotate_angle_tar', pred_window_rotate_angle_tar)
            else:
                pred_window_rotate_angle_tar = None

            ## crop the target again
            crop_inputs_tar = crop_inputs_tar_raw
            cropped_outputs_tar = image_cropping_stn(cursor_position_loop_tar, crop_inputs_tar, image_size, curr_window_size_tar_pred,
                                                     raster_size=self.hps.raster_size,
                                                     rotation_angle=pred_window_rotate_angle_tar)
            curr_patch_input_tar = cropped_outputs_tar[:, :, :, 0:1]  # (N, raster_size, raster_size, 1), [0.0-stroke, 1.0-BG]
            curr_patch_canvas_tar = cropped_outputs_tar[:, :, :, 1:2]  # (N, raster_size, raster_size, 1), [0.0-stroke, 1.0-BG]

            curr_patch_input_tar = normalize_image_m1to1(curr_patch_input_tar)
            curr_patch_canvas_tar = normalize_image_m1to1(curr_patch_canvas_tar)
            # (N, raster_size, raster_size, 1), [-1.0-stroke, 1.0-BG]

            #################### stage-1: end point encoder ####################
            # if seq_i == 1:
            #     prev_input_x_end = self.build_encoder_end(
            #         curr_patch_input_ref, curr_patch_canvas_ref, curr_patch_segment_ref, curr_patch_input_tar_temp, curr_patch_canvas_tar_temp)  # (N, z_size)
            # else:
            prev_input_x_end = self.build_encoder_end(
                curr_patch_input_ref, curr_patch_canvas_ref, curr_patch_segment_ref, curr_patch_input_tar, curr_patch_canvas_tar)  # (N, z_size)

            if self.hps.hidden_states_zero:
                prev_state_end = ((1 - starting_state).squeeze(dim=-1) * next_state_end[0],
                                  (1 - starting_state).squeeze(dim=-1) * next_state_end[1],
                                  (1 - starting_state).squeeze(dim=-1) * next_state_end[2],
                                  (1 - starting_state).squeeze(dim=-1) * next_state_end[3])
            else:
                prev_state_end = next_state_end
            h_output_end, next_state_end = self.build_decoder_end(prev_input_x_end, prev_state_end)
            # h_output: (N, n_out=2), next_state_end: (N, dec_rnn_size * 3)

            x0y0_global = torch.mul(cursor_position_loop_tar, image_size).squeeze(dim=1)  # (N, 2), in full size

            x3y3_rot = torch.tanh(h_output_end)  # (N, 2), [-1.0, 1.0]
            if self.hps.transform_with_rotation:
                x3y3 = spatial_transform_reverse_point(x3y3_rot, pred_window_rotate_angle_tar)  # (N, 2), [-1.0+, 1.0+]
            else:
                x3y3 = x3y3_rot
            x3y3_offset_global = x3y3 * (curr_window_size_tar_pred.squeeze(dim=1) / 2.0)  # (N, 2)
            x3y3_global = x0y0_global + x3y3_offset_global  # (N, 2), in full size

            pseudo_curve_parameters = torch.stack([x0y0_global, x3y3_global.detach()], dim=1)  # (N, 2, 2), in full size
            curr_line_image_large = self.rendering_line_image(pseudo_curve_parameters, image_size)  # (N, H, W), [0.0-stroke, 1.0-BG]
            curr_line_image_large = curr_line_image_large.unsqueeze(dim=-1)  # (N, H, W, 1), [0.0-stroke, 1.0-BG]

            line_image_crop_inputs_tar = curr_line_image_large
            curr_patch_line_tar = image_cropping_stn(cursor_position_loop_tar, line_image_crop_inputs_tar, image_size, curr_window_size_tar_pred,
                                                     raster_size=self.hps.raster_size,
                                                     rotation_angle=pred_window_rotate_angle_tar)
            # (N, raster_size, raster_size, 1), [0.0-stroke, 1.0-BG]
            curr_patch_line_tar = normalize_image_m1to1(curr_patch_line_tar)
            # (N, raster_size, raster_size, 1), [-1.0-stroke, 1.0-BG]

            #################### stage-2: control points encoder ####################

            prev_input_x_ctrl = self.build_encoder_ctrl(curr_patch_input_ref, curr_patch_canvas_ref, curr_patch_segment_ref,
                                                        curr_patch_input_tar, curr_patch_canvas_tar, curr_patch_line_tar)  # (N, z_size)
            if self.hps.hidden_states_zero:
                prev_state_ctrl = ((1 - starting_state).squeeze(dim=-1) * next_state_ctrl[0],
                                   (1 - starting_state).squeeze(dim=-1) * next_state_ctrl[1],
                                   (1 - starting_state).squeeze(dim=-1) * next_state_ctrl[2],
                                   (1 - starting_state).squeeze(dim=-1) * next_state_ctrl[3])
            else:
                prev_state_ctrl = next_state_ctrl
            h_output_ctrl, next_state_ctrl = self.build_decoder_ctrl(prev_input_x_ctrl, prev_state_ctrl)
            # h_output: (N, n_out=4), next_state_ctrl: (N, dec_rnn_size * 3)

            o_segment_params = torch.tanh(h_output_ctrl)  # (N, 4), [-1.0, 1.0]

            # convert the curve parameters into global coordinates
            x1y1_rot, x2y2_rot = o_segment_params[:, 0:2], o_segment_params[:, 2:4]  # (N, 2), [-1.0, 1.0]
            if self.hps.transform_with_rotation:
                x1y1 = spatial_transform_reverse_point(x1y1_rot, pred_window_rotate_angle_tar)  # (N, 2), [-1.0+, 1.0+]
                x2y2 = spatial_transform_reverse_point(x2y2_rot, pred_window_rotate_angle_tar)  # (N, 2), [-1.0+, 1.0+]
            else:
                x1y1 = x1y1_rot
                x2y2 = x2y2_rot

            x1y1_offset_global = x1y1 * (curr_window_size_tar_pred.squeeze(dim=1) / 2.0)  # (N, 2)
            x1y1_global = x0y0_global + x1y1_offset_global  # (N, 2), in full size

            x2y2_offset_global = x2y2 * (curr_window_size_tar_pred.squeeze(dim=1) / 2.0)  # (N, 2)
            x2y2_global = x0y0_global + x2y2_offset_global  # (N, 2), in full size

            curve_parameters = [x0y0_global, x1y1_global, x2y2_global, x3y3_global]
            curve_parameters = torch.stack(curve_parameters, dim=1)  # (N, 4, 2), in full size
            segment_params_list.append(curve_parameters)

            ## rendering
            curr_curve_image_large = self.rendering_curve_image(curve_parameters, image_size)  # (N, H, W), [0.0-stroke, 1.0-BG]
            curr_curve_image_large = 1.0 - curr_curve_image_large  # (N, H, W), [0.0-BG, 1.0-stroke]
            filter_curr_curve_image = curr_curve_image_large  # (N, H, W), [0.0-BG, 1.0-stroke]
            curr_canvas_tar_black = torch.add(curr_canvas_tar_black, filter_curr_curve_image)  # (N, H, W), [0.0-BG, 1.0-stroke]
            curr_canvas_tar_black = torch.clamp(curr_canvas_tar_black, 0.0, 1.0)  # (N, H, W), [0.0-BG, 1.0-stroke]

            # for debug visualization
            if self.model_mode == 'inference':
                # if seq_i in [1]:
                color_rgb = self.color_rgb_set[seq_i]  # (3) in [0.0, 1.0]
                color_rgb = np.reshape(color_rgb, (1, 1, 1, 3)).astype(np.float32)

                filter_curr_curve_image_ = filter_curr_curve_image.unsqueeze(dim=-1)  # (N, H, W, 1), [0.0-BG, 1.0-stroke]
                color_stroke = filter_curr_curve_image_ * torch.tensor(1.0 - color_rgb).float().cuda()
                curr_canvas_tar_rgb = curr_canvas_tar_rgb * (torch.sub(1.0, filter_curr_curve_image_)) \
                                      + color_stroke  # (N, H, W, 3), [0.0-BG, 1.0-stroke]

            # updating cursor_position
            cursor_position_loop_tar_inference = x3y3_global  # (N, 2), in full size
            cursor_position_loop_tar_inference = torch.max(cursor_position_loop_tar_inference, torch.tensor(0.0).cuda())
            cursor_position_loop_tar_inference = torch.min(cursor_position_loop_tar_inference, torch.tensor(image_size - 1).float().cuda())
            cursor_position_loop_tar_inference = torch.div(cursor_position_loop_tar_inference, image_size).unsqueeze(dim=1)  # (N, 1, 2), in [0.0, 1.0]

        segment_params = torch.stack(segment_params_list, dim=1)  # (N, seq_num, 4, 2), in full size

        return segment_params, curr_canvas_tar_black, curr_canvas_tar_rgb

    def build_encoder_transform(self, patch_input_ref, patch_input_tar, patch_input_tar_global):
        """
        :param patch_input_ref & patch_input_tar: (N, raster_size, raster_size, 1), [-1.0-stroke, 1.0-BG]
        :param patch_input_tar_global: (N, raster_size, raster_size, 1), [-1.0-stroke, 1.0-BG]
        :return:
        """
        # transform to nchw
        patch_inputs_ref = patch_input_ref  # (N, raster_size, raster_size, 1), [-1.0-stroke, 1.0-BG]
        patch_inputs_ref = patch_inputs_ref.permute(0, 3, 1, 2)  # (N, 1, raster_size, raster_size), [-1.0-stroke, 1.0-BG]
        patch_inputs_tar = patch_input_tar  # (N, raster_size, raster_size, 1), [-1.0-stroke, 1.0-BG]
        patch_inputs_tar = patch_inputs_tar.permute(0, 3, 1, 2)  # (N, 1, raster_size, raster_size), [-1.0-stroke, 1.0-BG]

        if self.hps.transform_use_global_info:
            patch_input_tar_global_resized = patch_input_tar_global
            patch_input_tar_global_resized = patch_input_tar_global_resized.permute(0, 3, 1, 2)  # (N, 1, raster_size, raster_size), [-1.0-stroke, 1.0-BG]
            patch_inputs_tar = torch.cat([patch_inputs_tar, patch_input_tar_global_resized], dim=1)  # (N, 2, raster_size, raster_size), [-1.0-stroke, 1.0-BG]

        if self.hps.enc_model_transform == 'combined':
            batch_input = torch.cat([patch_inputs_ref, patch_inputs_tar], dim=1)  # (N, 4, raster_size, raster_size), [-1.0-stroke, 1.0-BG]

            if self.hps.add_coordconv:
                batch_input = self.add_coords(batch_input)  # (N, in_dim + 2, in_H, in_W)
            output = self.encoder_transform(batch_input)  # (N, z_size)
        else:
            raise Exception('Unknown enc_model_transform:', self.hps.enc_model_transform)

        return output

    def build_encoder_end(self, patch_input_ref, patch_canvas_ref, patch_segment_ref, patch_input_tar, patch_canvas_tar):
        """
        :param 5 patch inputs: (N, raster_size, raster_size, 1), [-1.0-stroke, 1.0-BG]
        :return:
        """
        # transform to nchw
        patch_inputs_ref = torch.cat([patch_input_ref, patch_canvas_ref, patch_segment_ref], dim=-1)  # (N, raster_size, raster_size, 4), [-1.0-stroke, 1.0-BG]
        patch_inputs_ref = patch_inputs_ref.permute(0, 3, 1, 2)  # (N, 3, raster_size, raster_size), [-1.0-stroke, 1.0-BG]
        patch_inputs_tar = torch.cat([patch_input_tar, patch_canvas_tar], dim=-1)  # (N, raster_size, raster_size, 4), [-1.0-stroke, 1.0-BG]
        patch_inputs_tar = patch_inputs_tar.permute(0, 3, 1, 2)  # (N, 2, raster_size, raster_size), [-1.0-stroke, 1.0-BG]

        if 'separated' in self.hps.enc_model_tracing:
            if self.hps.add_coordconv:
                patch_inputs_ref = self.add_coords(patch_inputs_ref)  # (N, in_dim + 2, in_H, in_W)
                patch_inputs_tar = self.add_coords(patch_inputs_tar)  # (N, in_dim + 2, in_H, in_W)

            image_embedding = self.encoder_end(patch_inputs_ref, patch_inputs_tar)  # (N, z_size)
        else:
            raise Exception('Unknown enc_model_tracing:', self.hps.enc_model_tracing)

        return image_embedding

    def build_encoder_ctrl(self, patch_input_ref, patch_canvas_ref, patch_segment_ref,
                           patch_input_tar, patch_canvas_tar, patch_line_tar):
        """
        :param 6 patch inputs: (N, raster_size, raster_size, 1), [-1.0-stroke, 1.0-BG]
        :return:
        """
        # transform to nchw
        patch_inputs_ref = torch.cat([patch_input_ref, patch_canvas_ref, patch_segment_ref], dim=-1)  # (N, raster_size, raster_size, 3), [-1.0-stroke, 1.0-BG]
        patch_inputs_ref = patch_inputs_ref.permute(0, 3, 1, 2)  # (N, 3, raster_size, raster_size), [-1.0-stroke, 1.0-BG]
        patch_inputs_tar = torch.cat([patch_input_tar, patch_canvas_tar, patch_line_tar], dim=-1)  # (N, raster_size, raster_size, 3), [-1.0-stroke, 1.0-BG]
        patch_inputs_tar = patch_inputs_tar.permute(0, 3, 1, 2)  # (N, 3, raster_size, raster_size), [-1.0-stroke, 1.0-BG]

        if 'separated' in self.hps.enc_model_tracing:
            if self.hps.add_coordconv:
                patch_inputs_ref = self.add_coords(patch_inputs_ref)  # (N, in_dim + 2, in_H, in_W)
                patch_inputs_tar = self.add_coords(patch_inputs_tar)  # (N, in_dim + 2, in_H, in_W)

            image_embedding = self.encoder_ctrl(patch_inputs_ref, patch_inputs_tar)  # (N, z_size)
        else:
            raise Exception('Unknown enc_model_tracing:', self.hps.enc_model_tracing)

        return image_embedding

    def get_coordconv(self):
        xx_ones = torch.ones(self.hps.raster_size, dtype=torch.int32)  # e.g. (raster_size)
        xx_ones = xx_ones.unsqueeze(dim=-1)  # e.g. (raster_size, 1)
        xx_range = torch.arange(self.hps.raster_size, dtype=torch.int32)  # e.g. (raster_size)
        xx_range = xx_range.unsqueeze(0)  # e.g. (1, raster_size)

        xx_channel = torch.matmul(xx_ones, xx_range)  # e.g. (raster_size, raster_size)
        xx_channel = xx_channel.unsqueeze(0)  # e.g. (1, raster_size, raster_size)

        yy_ones = torch.ones(self.hps.raster_size, dtype=torch.int32)  # e.g. (raster_size)
        yy_ones = yy_ones.unsqueeze(0)  # e.g. (1, raster_size)
        yy_range = torch.arange(self.hps.raster_size, dtype=torch.int32)  # (raster_size)
        yy_range = yy_range.unsqueeze(-1)  # e.g. (raster_size, 1)

        yy_channel = torch.matmul(yy_range, yy_ones)  # e.g. (raster_size, raster_size)
        yy_channel = yy_channel.unsqueeze(0)  # e.g. (1, raster_size, raster_size)

        xx_channel = xx_channel.float() / (self.hps.raster_size - 1)
        yy_channel = yy_channel.float() / (self.hps.raster_size - 1)
        # xx_channel = xx_channel * 2 - 1  # [-1, 1]
        # yy_channel = yy_channel * 2 - 1

        # xx_channel = xx_channel.cuda()
        # yy_channel = yy_channel.cuda()

        ret = torch.cat([
            xx_channel,
            yy_channel,
        ], dim=0)  # (2, raster_size, raster_size)
        ret = ret.detach()

        return ret

    def add_coords(self, input_tensor):
        batch_size = input_tensor.size()[0]  # get N size
        coords = torch.unsqueeze(self.coordconv_input, dim=0).repeat(batch_size, 1, 1, 1)  # (N, 2, raster_size, raster_size)
        coords = coords.to(input_tensor.device)
        result = torch.cat([input_tensor, coords], dim=1)  # (N, C+2, raster_size, raster_size)
        return result

    def build_decoder_transform(self, dec_input, prev_state):
        """
        :param dec_input: (N, in_dim)
        :return:
        """
        h_output = self.decoder_transform(dec_input)
        next_state = None
        return h_output, next_state

    def build_decoder_end(self, dec_input, prev_state):
        """
        :param dec_input: (N, in_dim)
        :return:
        """
        h_output, next_state = self.decoder_end(dec_input, prev_state)
        # h_output: (N, n_out=6), next_state: (N, dec_rnn_size * 3)
        return h_output, next_state

    def build_decoder_ctrl(self, dec_input, prev_state):
        """
        :param dec_input: (N, in_dim)
        :return:
        """
        h_output, next_state = self.decoder_ctrl(dec_input, prev_state)
        # h_output: (N, n_out=6), next_state: (N, dec_rnn_size * 3)
        return h_output, next_state

    def rendering_curve_image(self, curve_params_batch, image_size):
        """
        :param curve_params_batch: (N, 4, 2)
        """
        batch_size, _, _ = curve_params_batch.shape
        curve_image_batch = []

        for batch_i in range(batch_size):
            shapes = []
            shape_groups = []

            curve_params = curve_params_batch[batch_i]  # (4, 2)

            num_control_points = torch.tensor([2])
            path = pydiffvg.Path(num_control_points=num_control_points,
                                 points=curve_params,
                                 is_closed=False,
                                 stroke_width=torch.tensor(self.stroke_thickness))
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
                         *scene_args)  # (H, W, 4), [0.0-stroke, 1.0-BG]
            curve_img = img[:, :, 0]  # (H, W), [0.0-stroke, 1.0-BG]
            curve_image_batch.append(curve_img)

        curve_image_batch = torch.stack(curve_image_batch, dim=0)  # (N, H, W), [0.0-stroke, 1.0-BG]
        return curve_image_batch

    def rendering_line_image(self, line_params_batch, image_size):
        """
        :param line_params_batch: (N, 2, 2)
        """
        batch_size, _, _ = line_params_batch.shape
        line_image_batch = []

        # no need to calculate gradient
        with torch.no_grad():
            for batch_i in range(batch_size):
                shapes = []
                shape_groups = []

                line_params = line_params_batch[batch_i]  # (2, 2)

                path = pydiffvg.Polygon(points=line_params,
                                        is_closed=False,
                                        stroke_width=torch.tensor(self.stroke_thickness))
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
                             *scene_args)  # (H, W, 4), [0.0-stroke, 1.0-BG]
                line_img = img[:, :, 0]  # (H, W), [0.0-stroke, 1.0-BG]
                line_image_batch.append(line_img)

        line_image_batch = torch.stack(line_image_batch, dim=0)  # (N, H, W), [0.0-stroke, 1.0-BG]
        return line_image_batch


class FullModel(object):
    def __init__(self, hps, valid_set):
        self.hps = hps
        self.valid_set = valid_set

        self.correspondence_model = Correspondence_Model(hps)
        self.generative_model = Generative_Model(hps, self.correspondence_model)
        self.perceptual_model = VGG_Slim()
        self.use_cuda = torch.cuda.is_available()

    def get_raster_loss(self, last_step_num, pred_imgs, gt_imgs, loss_type, return_map_pred, return_map_gt,
                        raster_perc_loss_layer, perc_loss_mean_list):
        perc_layer_losses_raw = []
        perc_layer_losses_norm = []

        if loss_type == 'l1':
            ras_cost = torch.mean(torch.abs(torch.sub(gt_imgs, pred_imgs)))  # ()
        elif loss_type == 'mse':
            ras_cost = torch.mean(torch.pow(torch.sub(gt_imgs, pred_imgs), 2))  # ()
        elif loss_type == 'perceptual':
            perc_loss_type = 'l1'  # [l1, mse]
            perc_layers = raster_perc_loss_layer

            for perc_layer in perc_layers:
                if perc_loss_type == 'l1':
                    perc_layer_loss = torch.mean(torch.abs(torch.sub(return_map_pred[perc_layer],
                                                                     return_map_gt[perc_layer])))  # ()
                elif perc_loss_type == 'mse':
                    perc_layer_loss = torch.mean(torch.pow(torch.sub(return_map_pred[perc_layer],
                                                                     return_map_gt[perc_layer]), 2))  # ()
                else:
                    raise NameError('Unknown perceptual loss type:', perc_loss_type)
                perc_layer_losses_raw.append(perc_layer_loss)

            for loop_i in range(len(perc_layers)):
                perc_relu_loss_raw = perc_layer_losses_raw[loop_i]  # ()
                curr_relu_mean = (perc_loss_mean_list[loop_i] * last_step_num + perc_relu_loss_raw) / (last_step_num + 1.0)
                relu_cost_norm = perc_relu_loss_raw / curr_relu_mean
                perc_layer_losses_norm.append(relu_cost_norm)

            perc_layer_losses_raw = torch.stack(perc_layer_losses_raw, dim=0)  # (n_layer)
            perc_layer_losses_norm = torch.stack(perc_layer_losses_norm, dim=0)  # (n_layer)

            if self.hps.perc_loss_fuse_type == 'max':
                ras_cost = torch.max(perc_layer_losses_norm)
            elif self.hps.perc_loss_fuse_type == 'add':
                ras_cost = torch.mean(perc_layer_losses_norm)
            elif self.hps.perc_loss_fuse_type == 'raw_add':
                ras_cost = torch.mean(perc_layer_losses_raw)
            else:
                raise NameError('Unknown perc_loss_fuse_type:', self.hps.perc_loss_fuse_type)
        else:
            raise NameError('Unknown loss type:', loss_type)

        return ras_cost, perc_layer_losses_raw, perc_layer_losses_norm

    def load_weights(self, pre_train_path, network):
        pretrained_dict = torch.load(pre_train_path)
        model_dict = network.state_dict()
        # print('pretrained_dict')
        # print(pretrained_dict.keys())
        # print('model_dict')
        # print(model_dict.keys())

        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        network.load_state_dict(model_dict)
        print('Loaded', pre_train_path)

    def print_model_variables(self, param_list, model_name):
        print('-' * 100)
        print('#', model_name)
        count_t_vars = 0
        for name, param in param_list:
            num_param = np.prod(list(param.size()))
            count_t_vars += num_param
            print('%s | shape: %s | num_param: %i |' % (name, str(param.size()), num_param), 'requires_grad:', param.requires_grad)
        print('Total trainable variables of %s: %i.' % (model_name, count_t_vars))
        # print('-' * 100)
        return count_t_vars

    def evaluate(self, load_trained_weights=False):
        print('-' * 100)
        print('Evaluation begins ...')

        if load_trained_weights:
            print('-' * 100)
            correspondence_transform_trained_weight_path = os.path.join(self.hps.trained_models_dir, 'point_matching_model/transform_module', 'sketch_transform_30000.pkl')
            correspondence_trained_weight_path = os.path.join(self.hps.trained_models_dir, 'point_matching_model', 'sketch_correspondence_50000.pkl')
            transform_tracing_trained_weight_path = os.path.join(self.hps.trained_models_dir, 'stroke_tracing_model', 'sketch_tracing_30000.pkl')
            self.load_weights(correspondence_transform_trained_weight_path, self.correspondence_model)
            self.load_weights(correspondence_trained_weight_path, self.correspondence_model)
            self.load_weights(transform_tracing_trained_weight_path, self.generative_model)
            self.load_weights(self.hps.perceptual_model_path, self.perceptual_model)
            print('-' * 100)

            if self.use_cuda:
                self.correspondence_model = self.correspondence_model.cuda()
                self.generative_model = self.generative_model.cuda()
                self.perceptual_model = self.perceptual_model.cuda()

        print('## All variables:')
        corr_num_param = self.print_model_variables(self.correspondence_model.named_parameters(), 'Correspondence_Model')
        gen_num_param = self.print_model_variables(self.generative_model.named_parameters(), 'Generative_Model')
        vgg_num_param = self.print_model_variables(self.perceptual_model.named_parameters(), 'Perceptual model')
        total_num_param = corr_num_param + gen_num_param
        print('Total trainable variables %i.' % total_num_param)

        self.correspondence_model.eval()
        self.generative_model.eval()
        self.perceptual_model.eval()

        self.valid_set.example_num = 1000
        batch_num = self.valid_set.example_num
        print('batch_num:', batch_num)

        perc_score_set = []
        endpoint_error_set = []
        allpoint_error_set = []

        with torch.no_grad():
            for batch_i in range(batch_num):
                print('batch_i', batch_i, '/', batch_num)
                reference_images, target_images, reference_segment_images, reference_dot_images_patch, \
                    reference_endpoints, starting_states, base_window_size, target_gt_points = \
                    self.valid_set.get_batch(self.use_cuda, batch_idx=batch_i)
                # reference_images: (N, H, W, 1), [0-stroke, 1-BG]
                # target_images: (N, H, W, 1), [0-stroke, 1-BG]
                # reference_segment_images: (N, seq_num, H, W), [0-stroke, 1-BG]
                # reference_dot_images_patch: (N, H_c, W_c), [0-stroke, 1-BG]
                # reference_endpoints: (N, seq_num, 2), in [0.0, 1.0]
                # starting_states: (N, seq_num), {1.0, 0.0}
                # base_window_size: (N, seq_num), in [0.0, 1.0]

                target_gt_points = target_gt_points.cpu().data.numpy()
                # target_gt_points:  (N, seq_num, 4, 2), [0.0, 1.0], numpy

                image_size = target_images.size()[1]
                seq_num = reference_segment_images.size()[1]

                raster_images_pred, _, params_pred = \
                    self.generative_model(seq_num=seq_num,
                                          reference_images=reference_images, target_images=target_images,
                                          reference_segment_images=reference_segment_images,
                                          reference_dot_images_patch=reference_dot_images_patch,
                                          endpoints_pos_ref=reference_endpoints,
                                          starting_states=starting_states,
                                          base_window_size=base_window_size,
                                          model_mode='eval', image_size=image_size, image_idx=batch_i)
                # raster_images_pred: (N, H, W), [0.0-stroke, 1.0-BG]
                # params_pred: (N, seq_num, 4, 2), in full size

                perc_map_pred = self.perceptual_model(raster_images_pred)
                perc_map_gt = self.perceptual_model(target_images.squeeze(dim=-1))

                _, perc_relu_losses_raw, _ = \
                    self.get_raster_loss(0, raster_images_pred, target_images.squeeze(dim=-1),
                                         loss_type=self.hps.raster_loss_base_type,
                                         return_map_pred=perc_map_pred, return_map_gt=perc_map_gt,
                                         raster_perc_loss_layer=self.hps.perc_loss_layers,
                                         perc_loss_mean_list=[0.0 for _ in range(len(self.hps.perc_loss_layers))])
                # perc_relu_losses_raw: (n_layer)

                # Perceptual score (PS): take the layer relu3_3 only
                perc_score = perc_relu_losses_raw[self.hps.perc_loss_layers.index('ReLU3_3')]
                perc_score = perc_score.cpu().data.numpy()
                perc_score_set.append(perc_score)

                # Endpoint error (EPE):
                params_norm_pred = params_pred.cpu().data.numpy() / float(image_size)  # (N, seq_num, 4, 2), [0.0, 1.0]

                endpoint_params_norm_pred = np.stack([params_norm_pred[:, :, 0, :], params_norm_pred[:, :, -1, :]], axis=2)   # (N, seq_num, 2, 2), [0.0, 1.0]
                endpoint_params_norm_gt = np.stack([target_gt_points[:, :, 0, :], target_gt_points[:, :, -1, :]], axis=2)  # (N, seq_num, 2, 2), [0.0, 1.0]
                endpoint_error_ = np.sqrt(np.sum(np.power(endpoint_params_norm_pred - endpoint_params_norm_gt, 2), axis=-1))  # (N, seq_num, 2)
                endpoint_error_ = np.sum(endpoint_error_, axis=-1)  # (N, seq_num)
                endpoint_error = np.mean(endpoint_error_, axis=-1)  # (N)
                endpoint_error_set.append(endpoint_error)

                # All point error (APE):
                allpoint_params_norm_pred = params_norm_pred  # (N, seq_num, 4, 2), [0.0, 1.0]
                allpoint_params_norm_gt = target_gt_points  # (N, seq_num, 4, 2), [0.0, 1.0]
                allpoint_error_ = np.sqrt(np.sum(np.power(allpoint_params_norm_pred - allpoint_params_norm_gt, 2), axis=-1))  # (N, seq_num, 4)
                allpoint_error_ = np.sum(allpoint_error_, axis=-1)  # (N, seq_num)
                allpoint_error = np.mean(allpoint_error_, axis=-1)  # (N)
                allpoint_error_set.append(allpoint_error)

            perc_score_avg = np.sum(perc_score_set) / float(self.valid_set.example_num)

            endpoint_error_set = np.concatenate(endpoint_error_set)
            endpoint_error_avg = np.mean(endpoint_error_set)

            allpoint_error_set = np.concatenate(allpoint_error_set)
            allpoint_error_avg = np.mean(allpoint_error_set)

            print('Perceptual score (PS):', perc_score_avg * 100.0, 'e-2')
            print('Endpoint error (EPE):', endpoint_error_avg * 100.0, 'e-2')
            print('All point error (APE):', allpoint_error_avg * 100.0, 'e-2')

    def inference(self, save_root, img_sequence):
        pred_black_save_root = os.path.join(save_root, 'raster')
        pred_rgb_save_root = os.path.join(save_root, 'rgb')
        pred_rgb_wobg_save_root = os.path.join(save_root, 'rgb-wo-bg')
        pred_params_save_root = os.path.join(save_root, 'parameter')
        os.makedirs(pred_black_save_root, exist_ok=True)
        os.makedirs(pred_rgb_save_root, exist_ok=True)
        os.makedirs(pred_rgb_wobg_save_root, exist_ok=True)
        os.makedirs(pred_params_save_root, exist_ok=True)

        print('Inference begins ...')

        correspondence_transform_trained_weight_path = os.path.join(self.hps.trained_models_dir, 'point_matching_model/transform_module', 'sketch_transform_30000.pkl')
        correspondence_trained_weight_path = os.path.join(self.hps.trained_models_dir, 'point_matching_model', 'sketch_correspondence_50000.pkl')
        transform_tracing_trained_weight_path = os.path.join(self.hps.trained_models_dir, 'stroke_tracing_model', 'sketch_tracing_30000.pkl')

        self.load_weights(correspondence_transform_trained_weight_path, self.correspondence_model)
        self.load_weights(correspondence_trained_weight_path, self.correspondence_model)
        self.load_weights(transform_tracing_trained_weight_path, self.generative_model)
        print('-' * 100)

        # print('## All variables:')
        # corr_num_param = self.print_model_variables(self.correspondence_model.named_parameters(), 'Correspondence_Model')
        # gen_num_param = self.print_model_variables(self.generative_model.named_parameters(), 'Generative_Model')
        # total_num_param = corr_num_param + gen_num_param
        # print('Total trainable variables %i.' % total_num_param)

        if self.use_cuda:
            self.correspondence_model = self.correspondence_model.cuda()
            self.generative_model = self.generative_model.cuda()

        self.correspondence_model.eval()
        self.generative_model.eval()

        with torch.no_grad():
            for pair_i in range(len(img_sequence) - 1):
                reference_img_name = img_sequence[pair_i]
                target_img_name = img_sequence[pair_i + 1]
                print('reference_img_name / target_img_name:', reference_img_name, '/', target_img_name)
                target_id = target_img_name[:target_img_name.find('.')]
                manual_reference = pair_i == 0

                reference_images, target_images, reference_segment_images, reference_dot_images_patch, \
                    reference_endpoints, starting_states, base_window_size, _ = \
                    self.valid_set.get_batch(self.use_cuda, reference_img_name, target_img_name, manual_reference)
                # reference_images: (N, H, W, 1), [0-stroke, 1-BG]
                # target_images: (N, H, W, 1), [0-stroke, 1-BG]
                # reference_segment_images: (N, seq_num, H, W), [0-stroke, 1-BG]
                # reference_dot_images_patch: (N, H_c, W_c), [0-stroke, 1-BG]
                # reference_endpoints: (N, seq_num, 2), in [0.0, 1.0]
                # starting_states: (N, seq_num), {1.0, 0.0}
                # base_window_size: (N, seq_num), in [0.0, 1.0]

                image_size = target_images.size()[1]
                seq_num = reference_segment_images.size()[1]

                raster_images_pred, raster_images_pred_rgb, params_pred = \
                    self.generative_model(seq_num=seq_num,
                                          reference_images=reference_images, target_images=target_images,
                                          reference_segment_images=reference_segment_images,
                                          reference_dot_images_patch=reference_dot_images_patch,
                                          endpoints_pos_ref=reference_endpoints,
                                          starting_states=starting_states,
                                          base_window_size=base_window_size,
                                          model_mode='inference', image_size=image_size)
                # raster_images_pred: (N, H, W), [0.0-stroke, 1.0-BG]
                # params_pred: (N, seq_num, 4, 2), in full size

                target_images_bg = target_images.cpu().data.numpy()

                img_i = 0
                params_pred_np = params_pred[img_i].cpu().data.numpy()  # (seq_num, 4, 2), in full size
                starting_states_np = starting_states[img_i].cpu().data.numpy()  # (seq_num), {1.0, 0.0}
                params_pred_list = seq_params_to_list(params_pred_np, starting_states_np)  # list of (N_point, 2), in image size
                params_pred_save_npz_path = os.path.join(pred_params_save_root, target_id + '.npz')
                np.savez(params_pred_save_npz_path, strokes_data=params_pred_list)

                raster_images_pred = raster_images_pred.cpu().data.numpy()
                raster_images_pred = (np.array(raster_images_pred[img_i]) * 255.0).astype(np.uint8)
                pred_save_path = os.path.join(pred_black_save_root, target_id + '.png')
                raster_images_pred = Image.fromarray(raster_images_pred, 'L')
                raster_images_pred.save(pred_save_path, 'PNG')

                raster_images_pred_rgb = raster_images_pred_rgb.cpu().data.numpy()
                raster_images_pred_rgb = (np.array(raster_images_pred_rgb[img_i]) * 255.0).astype(np.uint8)
                pred_rgb_wo_bg_save_path = os.path.join(pred_rgb_wobg_save_root, target_id + '.png')
                raster_images_pred_rgb_png = Image.fromarray(raster_images_pred_rgb, 'RGB')
                raster_images_pred_rgb_png.save(pred_rgb_wo_bg_save_path, 'PNG')

                raster_images_pred_rgb_bg = target_images_bg[img_i, :, :, 0] * 255.0  # (H, W), [0-stroke, 255-BG]
                raster_images_pred_rgb_bg = np.tile(np.expand_dims(raster_images_pred_rgb_bg, axis=-1), (1, 1, 3))
                raster_images_pred_rgb_bg = 255 - (255 - raster_images_pred_rgb_bg) * 0.2
                rgb_mask = (raster_images_pred_rgb != 255).any(-1)
                raster_images_pred_rgb_bg[rgb_mask] = raster_images_pred_rgb[rgb_mask]
                raster_images_pred_rgb_bg = raster_images_pred_rgb_bg.astype(np.uint8)
                pred_rgb_save_path = os.path.join(pred_rgb_save_root, target_id + '.png')
                raster_images_pred_rgb_bg = Image.fromarray(raster_images_pred_rgb_bg, 'RGB')
                raster_images_pred_rgb_bg.save(pred_rgb_save_path, 'PNG')
