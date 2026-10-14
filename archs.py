import torch
from torch import nn
import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
from utils import *

import timm
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import types
import math
from abc import ABCMeta, abstractmethod
# from mmcv.cnn import ConvModule
from pdb import set_trace as st

# from kan import KANLinear, KAN
from torch.nn import init
# from fun_kan import FUNKAN,FUNKAN_HybridAttention,FUNKAN_ChannelAttention,FUNKAN_SpatialAttention
from sg_kan import SGKAN_Hybrid, SGKAN_SpatialOnly, SGKAN_ChannelOnly

class KANLayer(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., no_kan=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.dim = in_features
        
        grid_size=5
        spline_order=3
        scale_noise=0.1
        scale_base=1.0
        scale_spline=1.0
        base_activation=torch.nn.SiLU
        grid_eps=0.02
        grid_range=[-1, 1]

        if not no_kan:
            self.fc1 = KANLinear(
                        in_features,
                        hidden_features,
                        grid_size=grid_size,
                        spline_order=spline_order,
                        scale_noise=scale_noise,
                        scale_base=scale_base,
                        scale_spline=scale_spline,
                        base_activation=base_activation,
                        grid_eps=grid_eps,
                        grid_range=grid_range,
                    )
            self.fc2 = KANLinear(
                        hidden_features,
                        out_features,
                        grid_size=grid_size,
                        spline_order=spline_order,
                        scale_noise=scale_noise,
                        scale_base=scale_base,
                        scale_spline=scale_spline,
                        base_activation=base_activation,
                        grid_eps=grid_eps,
                        grid_range=grid_range,
                    )
            self.fc3 = KANLinear(
                        hidden_features,
                        out_features,
                        grid_size=grid_size,
                        spline_order=spline_order,
                        scale_noise=scale_noise,
                        scale_base=scale_base,
                        scale_spline=scale_spline,
                        base_activation=base_activation,
                        grid_eps=grid_eps,
                        grid_range=grid_range,
                    )
            # # TODO   
            # self.fc4 = KANLinear(
            #             hidden_features,
            #             out_features,
            #             grid_size=grid_size,
            #             spline_order=spline_order,
            #             scale_noise=scale_noise,
            #             scale_base=scale_base,
            #             scale_spline=scale_spline,
            #             base_activation=base_activation,
            #             grid_eps=grid_eps,
            #             grid_range=grid_range,
            #         )   

        else:
            self.fc1 = nn.Linear(in_features, hidden_features)
            self.fc2 = nn.Linear(hidden_features, out_features)
            self.fc3 = nn.Linear(hidden_features, out_features)

        # TODO
        # self.fc1 = nn.Linear(in_features, hidden_features)


        self.dwconv_1 = DW_bn_relu(hidden_features)
        self.dwconv_2 = DW_bn_relu(hidden_features)
        self.dwconv_3 = DW_bn_relu(hidden_features)

        # # TODO
        # self.dwconv_4 = DW_bn_relu(hidden_features)
    
        self.drop = nn.Dropout(drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
    

    def forward(self, x, H, W):
        # pdb.set_trace()
        B, N, C = x.shape

        x = self.fc1(x.reshape(B*N,C)) #torch.Size([2048, 160])
        x = x.reshape(B,N,C).contiguous() #torch.Size([8, 256, 160])
        x = self.dwconv_1(x, H, W)#torch.Size([8, 256, 160])
        x = self.fc2(x.reshape(B*N,C))
        x = x.reshape(B,N,C).contiguous()
        x = self.dwconv_2(x, H, W)
        x = self.fc3(x.reshape(B*N,C))
        x = x.reshape(B,N,C).contiguous()
        x = self.dwconv_3(x, H, W)

        # # TODO
        # x = x.reshape(B,N,C).contiguous()
        # x = self.dwconv_4(x, H, W)
    
        return x

class KANBlock(nn.Module):
    def __init__(self, dim, drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, no_kan=False):
        super().__init__()

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim)

        self.layer = KANLayer(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop, no_kan=no_kan)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = x + self.drop_path(self.layer(self.norm2(x), H, W))

        return x


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x

class DW_bn_relu(nn.Module):
    def __init__(self, dim=768):
        super(DW_bn_relu, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)
        self.bn = nn.BatchNorm2d(dim)
        self.relu = nn.ReLU()

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W) #torch.Size([8, 160, 16, 16])
        x = self.dwconv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = x.flatten(2).transpose(1, 2)

        return x

class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)

        return x, H, W


class ConvLayer(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(ConvLayer, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        return self.conv(input)

class D_ConvLayer(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(D_ConvLayer, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, padding=1),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        return self.conv(input)



class UKAN(nn.Module):
    def __init__(self, num_classes, input_channels=3, deep_supervision=False, img_size=224, patch_size=16, in_chans=3, embed_dims=[256, 320, 512], no_kan=False,
    drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm, depths=[1, 1, 1], **kwargs):
        super().__init__()

        kan_input_dim = embed_dims[0]

        self.encoder1 = ConvLayer(3, kan_input_dim//8)  
        self.encoder2 = ConvLayer(kan_input_dim//8, kan_input_dim//4)  
        self.encoder3 = ConvLayer(kan_input_dim//4, kan_input_dim)

        self.norm3 = norm_layer(embed_dims[1])
        self.norm4 = norm_layer(embed_dims[2])

        self.dnorm3 = norm_layer(embed_dims[1])
        self.dnorm4 = norm_layer(embed_dims[0])

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.block1 = nn.ModuleList([KANBlock(
            dim=embed_dims[1], 
            drop=drop_rate, drop_path=dpr[0], norm_layer=norm_layer
            )])

        self.block2 = nn.ModuleList([KANBlock(
            dim=embed_dims[2],
            drop=drop_rate, drop_path=dpr[1], norm_layer=norm_layer
            )])

        self.dblock1 = nn.ModuleList([KANBlock(
            dim=embed_dims[1], 
            drop=drop_rate, drop_path=dpr[0], norm_layer=norm_layer
            )])

        self.dblock2 = nn.ModuleList([KANBlock(
            dim=embed_dims[0], 
            drop=drop_rate, drop_path=dpr[1], norm_layer=norm_layer
            )])

        self.patch_embed3 = PatchEmbed(img_size=img_size // 4, patch_size=3, stride=2, in_chans=embed_dims[0], embed_dim=embed_dims[1])
        self.patch_embed4 = PatchEmbed(img_size=img_size // 8, patch_size=3, stride=2, in_chans=embed_dims[1], embed_dim=embed_dims[2])

        self.decoder1 = D_ConvLayer(embed_dims[2], embed_dims[1])  
        self.decoder2 = D_ConvLayer(embed_dims[1], embed_dims[0])  
        self.decoder3 = D_ConvLayer(embed_dims[0], embed_dims[0]//4) 
        self.decoder4 = D_ConvLayer(embed_dims[0]//4, embed_dims[0]//8)
        self.decoder5 = D_ConvLayer(embed_dims[0]//8, embed_dims[0]//8)

        self.final = nn.Conv2d(embed_dims[0]//8, num_classes, kernel_size=1)
        self.soft = nn.Softmax(dim =1)

    def forward(self, x):
        
        B = x.shape[0]
        ### Encoder
        ### Conv Stage

        ### Stage 1
        out = F.relu(F.max_pool2d(self.encoder1(x), 2, 2))
        t1 = out
        ### Stage 2
        out = F.relu(F.max_pool2d(self.encoder2(out), 2, 2))
        t2 = out
        ### Stage 3
        out = F.relu(F.max_pool2d(self.encoder3(out), 2, 2))
        t3 = out

        ### Tokenized KAN Stage
        ### Stage 4

        out, H, W = self.patch_embed3(out)
        for i, blk in enumerate(self.block1):
            out = blk(out, H, W)
        out = self.norm3(out) #[8, 256, 160]
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        t4 = out #torch.Size([8, 160, 16, 16])

        ### Bottleneck

        out, H, W= self.patch_embed4(out) #torch.Size([8, 64, 256])
        for i, blk in enumerate(self.block2):
            out = blk(out, H, W)
        out = self.norm4(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous() #torch.Size([8, 256, 8, 8])

        ### Stage 4
        out = F.relu(F.interpolate(self.decoder1(out), scale_factor=(2,2), mode ='bilinear'))

        out = torch.add(out, t4)
        _, _, H, W = out.shape
        out = out.flatten(2).transpose(1,2)
        for i, blk in enumerate(self.dblock1):
            out = blk(out, H, W)

        ### Stage 3
        out = self.dnorm3(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        out = F.relu(F.interpolate(self.decoder2(out),scale_factor=(2,2),mode ='bilinear'))
        out = torch.add(out,t3)
        _,_,H,W = out.shape
        out = out.flatten(2).transpose(1,2)
        
        for i, blk in enumerate(self.dblock2):
            out = blk(out, H, W)

        out = self.dnorm4(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        out = F.relu(F.interpolate(self.decoder3(out),scale_factor=(2,2),mode ='bilinear'))
        out = torch.add(out,t2)
        out = F.relu(F.interpolate(self.decoder4(out),scale_factor=(2,2),mode ='bilinear'))
        out = torch.add(out,t1)
        out = F.relu(F.interpolate(self.decoder5(out),scale_factor=(2,2),mode ='bilinear'))

        return self.final(out)

class HiFuse_SGKAN(nn.Module):
    def __init__(self, num_classes, input_channels=3, deep_supervision=False, img_size=224, patch_size=16, in_chans=3, embed_dims=[256, 320, 512], no_kan=False,
    drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm, depths=[1, 1, 1], global_dims=[8, 16, 32, 64, 128], **kwargs):
        super().__init__()
        embed_dims=[128, 160, 256]
        global_dims=[8, 16, 32, 64, 128]
        kan_input_dim = embed_dims[0]

        self.encoder1 = ConvLayer(3, kan_input_dim//8)  
        self.encoder2 = ConvLayer(kan_input_dim//8, kan_input_dim//4)  
        self.encoder3 = ConvLayer(kan_input_dim//4, kan_input_dim//2)
        self.encoder4 = ConvLayer(kan_input_dim//2, kan_input_dim)
        
        self.backbone = nn.ModuleList([SGKAN_Hybrid(kan_input_dim, kan_input_dim, **kwargs) for _ in range(3)])
        
        # self.decoder1 = D_ConvLayer(embed_dims[2], embed_dims[1])  
        self.decoder2 = D_ConvLayer(kan_input_dim, kan_input_dim//2)  
        self.decoder3 = D_ConvLayer(kan_input_dim//2 + global_dims[2], kan_input_dim//4)  
        self.decoder4 = D_ConvLayer(kan_input_dim//4 + global_dims[1], kan_input_dim//8)
        self.decoder5 = D_ConvLayer(kan_input_dim//8 + global_dims[0], kan_input_dim//8)

        self.final = nn.Conv2d(kan_input_dim//8, num_classes, kernel_size=1)
        
        # 浅层（SBE浅层边界增强）
        # 深层（DCA 深层全局上下文融合模块）
        self.gsb1 = SBE(in_ch=kan_input_dim//8, out_ch=global_dims[0], drop_path=drop_path_rate*0.3)          
        self.gsb2 = SBE(in_ch=kan_input_dim//4, out_ch=global_dims[1], drop_path=drop_path_rate*0.3)
        self.gsb3 = DCA(in_ch=kan_input_dim//2, out_ch=global_dims[2], drop_path=drop_path_rate)
        
    def forward(self, x):
        
        B = x.shape[0]
        ### Stage 1
        out = F.relu(F.max_pool2d(self.encoder1(x), 2, 2)) #torch.Size([8, 16, 128, 128])
        t1_slb = self.gsb1(out) #torch.Size([8, 8, 128, 128])
        ### Stage 2
        out = F.relu(F.max_pool2d(self.encoder2(out), 2, 2)) #torch.Size([8, 32, 64, 64])
        t2_slb = self.gsb2(out) #torch.Size([8, 16, 64, 64])
        ### Stage 3
        out = F.relu(F.max_pool2d(self.encoder3(out), 2, 2)) #torch.Size([8, 64, 32, 32])
        t3_lgc = self.gsb3(out) #torch.Size([8, 32, 32, 32])

        out = F.relu(F.max_pool2d(self.encoder4(out), 2, 2)) #torch.Size([8, 128, 16, 16])
        
        # 2. Backbone.
        for layer in self.backbone:
            out = layer(out) #[8, 128, 16, 16]
            
        out = F.relu(F.interpolate(self.decoder2(out), scale_factor=2, mode='bilinear'))#[8, 64, 32, 32])
        out = torch.cat([out, t3_lgc], dim=1)  # 融合全局特征 # torch.Size([8, 96, 32, 32])
        
        out = F.relu(F.interpolate(self.decoder3(out),scale_factor=(2,2),mode ='bilinear')) #torch.Size([8, 32, 64, 64])
        out = torch.cat([out,t2_slb],dim=1) #torch.Size([8, 48, 64, 64])
        
        out = F.relu(F.interpolate(self.decoder4(out),scale_factor=(2,2),mode ='bilinear')) #torch.Size([8, 16, 128, 128])
        out = torch.cat([out,t1_slb],dim=1) #torch.Size([8, 24, 128, 128])
        
        out = F.relu(F.interpolate(self.decoder5(out),scale_factor=(2,2),mode ='bilinear')) #torch.Size([8, 16, 256, 256])

        return self.final(out)
class LightConvTextureBranch(nn.Module):
    """
    轻量化卷积纹理分支（SBE适配）
    特点：深度可分离卷积（参数减少8倍）+ 多感受野（1x1+3x3）+ 轻量化注意力
    """
    def __init__(self, in_ch, out_ch):
        super().__init__()
        # 多感受野特征提取（1x1捕捉点纹理，3x3捕捉局部纹理）
        self.conv1x1 = nn.Conv2d(in_ch, out_ch//2, 1, padding=0, bias=False)
        self.depthwise_conv3x3 = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, padding=1, groups=in_ch, bias=False),  # 深度卷积
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch//2, 1, bias=False)  # 点卷积
        )
        
        # 轻量化通道注意力（校准纹理特征）
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_ch, out_ch//4, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch//4, out_ch, 1, bias=False),
            nn.Sigmoid()
        )
        
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # 多感受野融合
        feat1 = self.conv1x1(x)
        feat3 = self.depthwise_conv3x3(x)
        feat = torch.cat([feat1, feat3], dim=1)  # (B, out_ch, H, W)
        
        # 注意力校准
        attn = self.attention(feat)
        feat = feat * attn
        
        # 归一化+激活
        return self.relu(self.bn(feat))
class SBE(nn.Module):
    def __init__(self, in_ch, out_ch, drop_path=0., vis_mode=False):
        super().__init__()
        self.vis_mode = vis_mode
        self.out_ch = out_ch
        self.in_ch = in_ch

        # 校验分组卷积条件
        if in_ch % out_ch != 0:
            raise ValueError(f"输入通道数{in_ch}必须能被输出通道数{out_ch}整除（分组数=out_ch）")
        
        # 纹理分支
        self.local_branch = LightConvTextureBranch(in_ch, out_ch)

        # 多方向边缘核（适配分组卷积）
        edge_kernels = torch.tensor([
            [[-1, -1, -1], [2, 2, 2], [-1, -1, -1]],
            [[-1, 2, -1], [-1, 2, -1], [-1, 2, -1]],
            [[-1, -1, 2], [-1, 2, -1], [2, -1, -1]],
            [[2, -1, -1], [-1, 2, -1], [-1, -1, 2]]
        ], dtype=torch.float32)
        base_kernel = edge_kernels.mean(dim=0, keepdim=True)  # [1, 3, 3]
        base_kernel = base_kernel.unsqueeze(0).repeat(out_ch, in_ch//out_ch, 1, 1)  # [out_ch, in_ch//out_ch, 3, 3]
        self.edge_kernels = nn.ParameterList([
            nn.Parameter(base_kernel.clone(), requires_grad=True) for _ in range(4)
        ])

        # 自适应融合
        self.fusion_weight = nn.Sequential(
            nn.Conv2d(out_ch * 2, out_ch, 1, bias=False),
            nn.Sigmoid()
        )

        # 残差连接
        self.residual = nn.Conv2d(in_ch, out_ch, 1, bias=False) if in_ch != out_ch else nn.Identity()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        self.x_ori = x.detach()
        
        # 纹理特征
        tex_feat = self.local_branch(x)
        self.tex_feat = tex_feat.detach()
        
        # 多方向边缘特征
        edge_feats = []
        for kernel in self.edge_kernels:
            edge_feat = F.conv2d(
                x, 
                kernel, 
                padding=1, 
                groups=self.out_ch
            )
            edge_feats.append(edge_feat)
        edge_feat = torch.stack(edge_feats, dim=2).mean(dim=2)
        self.edge_feat = edge_feat.detach()
        
        # 融合
        fusion_feat = torch.cat([tex_feat, edge_feat], dim=1)
        fusion_weight = self.fusion_weight(fusion_feat)
        fused_feat = fusion_weight * tex_feat + (1 - fusion_weight) * edge_feat
        
        # 残差
        out = self.residual(x) + self.drop_path(fused_feat)
        self.fused_feat = out.detach()
        
        return out
# 2. 深层轻量模块（DCA）- 补全局上下文
class DCA(nn.Module):
    """深层上下文聚合模块：多尺度特征聚合 + 通道注意力"""
    def __init__(self, in_ch, out_ch, reduction=8, drop_path=0.):
        super().__init__()
        mid_ch = in_ch // reduction  # 通道压缩比例
        
        # 多尺度特征提取（不同膨胀率的卷积）
        self.multi_scale = nn.ModuleList([
            nn.Conv2d(in_ch, mid_ch, 3, padding=1, bias=False),      # 普通卷积
            nn.Conv2d(in_ch, mid_ch, 3, padding=2, dilation=2, bias=False)  # 膨胀卷积
        ])
        
        self.bn = nn.BatchNorm2d(mid_ch * 2)
        self.relu = nn.ReLU(inplace=True)
        
        # 通道注意力机制
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(mid_ch * 2, mid_ch, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, mid_ch * 2, 1, bias=False),
            nn.Sigmoid()
        )
        
        # 特征校准
        self.calibrate = nn.Conv2d(mid_ch * 2, out_ch, 1, bias=False)
        
        # 残差连接
        self.residual = nn.Conv2d(in_ch, out_ch, 1, bias=False) if in_ch != out_ch else nn.Identity()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        # 多尺度特征提取
        scale_feats = torch.cat([conv(x) for conv in self.multi_scale], dim=1)
        
        # 批归一化和激活
        scale_feats = self.relu(self.bn(scale_feats))
        
        # 通道注意力加权
        attn_weight = self.channel_attention(scale_feats)
        attended_feats = scale_feats * attn_weight
        
        # 特征校准
        calibrated = self.calibrate(attended_feats)
        
        # 残差连接 + DropPath
        return self.residual(x) + self.drop_path(calibrated)
