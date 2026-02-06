# Copyright (c) [2012]-[2021] Shanghai Yitu Technology Co., Ltd.
#
# This source code is licensed under the Clear BSD License
# LICENSE file in the root directory of this file
# All rights reserved.
"""
Borrow from timm(https://github.com/rwightman/pytorch-image-models)
"""
import torch
import torch.nn as nn
import numpy as np
from timm.models.layers import DropPath


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MutualAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.scale = qk_scale or head_dim ** -0.5

        self.rgb_q = nn.Linear(dim, dim, bias=qkv_bias)
        self.rgb_k = nn.Linear(dim, dim, bias=qkv_bias)
        self.rgb_v = nn.Linear(dim, dim, bias=qkv_bias)
        self.rgb_proj = nn.Linear(dim, dim)

        self.depth_q = nn.Linear(dim, dim, bias=qkv_bias)
        self.depth_k = nn.Linear(dim, dim, bias=qkv_bias)
        self.depth_v = nn.Linear(dim, dim, bias=qkv_bias)
        self.depth_proj = nn.Linear(dim, dim)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, rgb_fea, depth_fea):
        B, N, C = rgb_fea.shape

        rgb_q = self.rgb_q(rgb_fea).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        rgb_k = self.rgb_k(rgb_fea).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        rgb_v = self.rgb_v(rgb_fea).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        # q [B, nhead, N, C//nhead]

        depth_q = self.depth_q(depth_fea).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        depth_k = self.depth_k(depth_fea).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        depth_v = self.depth_v(depth_fea).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        # rgb branch
        rgb_attn = (rgb_q @ depth_k.transpose(-2, -1)) * self.scale
        rgb_attn = rgb_attn.softmax(dim=-1)
        rgb_attn = self.attn_drop(rgb_attn)

        rgb_fea = (rgb_attn @ depth_v).transpose(1, 2).reshape(B, N, C)
        rgb_fea = self.rgb_proj(rgb_fea)
        rgb_fea = self.proj_drop(rgb_fea)

        # depth branch
        depth_attn = (depth_q @ rgb_k.transpose(-2, -1)) * self.scale
        depth_attn = depth_attn.softmax(dim=-1)
        depth_attn = self.attn_drop(depth_attn)

        depth_fea = (depth_attn @ rgb_v).transpose(1, 2).reshape(B, N, C)
        depth_fea = self.depth_proj(depth_fea)
        depth_fea = self.proj_drop(depth_fea)

        return rgb_fea, depth_fea


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class MutualSelfBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)

        # mutual attention
        self.norm1_rgb_ma = norm_layer(dim)
        self.norm2_depth_ma = norm_layer(dim)
        self.mutualAttn = MutualAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.norm3_rgb_ma = norm_layer(dim)
        self.norm4_depth_ma = norm_layer(dim)
        self.mlp_rgb_ma = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.mlp_depth_ma = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        # rgb self attention
        self.norm1_rgb_sa = norm_layer(dim)
        self.selfAttn_rgb = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.norm2_rgb_sa = norm_layer(dim)
        self.mlp_rgb_sa = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        # depth self attention
        self.norm1_depth_sa = norm_layer(dim)
        self.selfAttn_depth = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.norm2_depth_sa = norm_layer(dim)
        self.mlp_depth_sa = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, rgb_fea, depth_fea):

        # mutual attention
        rgb_fea_fuse, depth_fea_fuse = self.drop_path(self.mutualAttn(self.norm1_rgb_ma(rgb_fea), self.norm2_depth_ma(depth_fea)))

        rgb_fea = rgb_fea + rgb_fea_fuse
        depth_fea = depth_fea + depth_fea_fuse

        rgb_fea = rgb_fea + self.drop_path(self.mlp_rgb_ma(self.norm3_rgb_ma(rgb_fea)))
        depth_fea = depth_fea + self.drop_path(self.mlp_depth_ma(self.norm4_depth_ma(depth_fea)))

        # rgb self attention
        rgb_fea = rgb_fea + self.drop_path(self.selfAttn_rgb(self.norm1_rgb_sa(rgb_fea)))
        rgb_fea = rgb_fea + self.drop_path(self.mlp_rgb_sa(self.norm2_rgb_sa(rgb_fea)))

        # depth self attention
        depth_fea = depth_fea + self.drop_path(self.selfAttn_depth(self.norm1_depth_sa(depth_fea)))
        depth_fea = depth_fea + self.drop_path(self.mlp_depth_sa(self.norm2_depth_sa(depth_fea)))

        return rgb_fea, depth_fea


def get_sinusoid_encoding(n_position, d_hid):
    ''' Sinusoid position encoding table '''

    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    return torch.FloatTensor(sinusoid_table).unsqueeze(0)
