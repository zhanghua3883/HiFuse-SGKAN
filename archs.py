import torch
import torch.nn as nn
import torch.nn.functional as F
from sg_kan import SGKAN_Hybrid


# --------------------------- Basic Convolution Block ---------------------------
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

    def forward(self, x):
        return self.conv(x)


# --------------------------- Upsampling Convolution Block ---------------------------
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

    def forward(self, x):
        return self.conv(x)


# --------------------------- DropPath (Stochastic Depth) ---------------------------
def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    def __init__(self, drop_prob=0.):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


# --------------------------- SBE: Shallow Boundary Enhancement ---------------------------
class SBE(nn.Module):
    """Shallow Boundary Enhancement: internal texture branch uses add instead of cat;
    outer texture + edge branch keeps concatenation."""

    def __init__(self, in_ch, out_ch, drop_path=0.):
        super().__init__()
        self.act = nn.ReLU(inplace=True)

        # Texture feature branch
        self.pw_conv = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.dw_conv = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, groups=out_ch, bias=False)
        self.pw_expand = nn.Conv2d(out_ch, out_ch, kernel_size=1, bias=False)
        self.bn_sep = nn.BatchNorm2d(out_ch)

        # Multi-directional edge detection depthwise convolutions
        assert out_ch % in_ch == 0, "SBE: out_ch must be divisible by in_ch for depthwise edge conv"
        self.edge_convs = nn.ModuleList()
        for kernel in self._init_edge_kernels():
            conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, groups=in_ch, bias=False)
            k4d = kernel.unsqueeze(0).unsqueeze(0).expand(out_ch, 1, 3, 3).contiguous()
            conv.weight.data.copy_(k4d)
            self.edge_convs.append(conv)
        self.edge_bn = nn.BatchNorm2d(out_ch)

        # Outer fusion still uses concatenation (2*out_ch channels), gating structure preserved
        self.gate = nn.Sequential(
            nn.Conv2d(out_ch * 2, out_ch * 2, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
        self.final_proj = nn.Conv2d(out_ch * 2, out_ch, kernel_size=1, bias=False)

        # Residual branch
        self.residual = nn.Conv2d(in_ch, out_ch, 1, bias=False) if in_ch != out_ch else nn.Identity()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def _init_edge_kernels(self):
        sobel_h = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        sobel_v = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        diag1 = torch.tensor([[-2, -1, 0], [-1, 0, 1], [0, 1, 2]], dtype=torch.float32)
        diag2 = torch.tensor([[0, 1, 2], [-1, 0, 1], [-2, -1, 0]], dtype=torch.float32)
        return [sobel_h, sobel_v, diag1, diag2]

    def forward(self, x):
        # Texture branch: internal fusion uses add instead of cat
        f1 = self.pw_conv(x)
        f_dw = self.dw_conv(f1)
        f_sep = self.pw_expand(f_dw)
        tex_fuse = torch.add(f1, f_sep)
        tex_feat = self.act(self.bn_sep(tex_fuse))

        # Edge detection branch
        edge_feats = [conv(x) for conv in self.edge_convs]
        edge_feat = sum(edge_feats) / 4.0
        edge_feat = self.act(self.edge_bn(edge_feat))

        # Outer core fusion: keep concatenation unchanged
        f_cat = torch.cat([tex_feat, edge_feat], dim=1)
        g = self.gate(f_cat)
        f_gated = g * f_cat
        f_gated = self.final_proj(f_gated)

        # Residual output
        out = self.residual(x) + self.drop_path(f_gated)
        return out


# --------------------------- DCA: Deep Context Aggregation ---------------------------
class DCA(nn.Module):
    """Deep Context Aggregation: dual-branch dilated convolution for multi-scale context,
    SE-style channel attention gating, and residual connection.
    Config: gamma=8 (channel attention reduction ratio), DropPath=0.5 (Encoder3 stage)."""

    def __init__(self, in_ch, out_ch, gamma=8, drop_path=0.):
        super().__init__()
        # Multi-scale dilated convolution branches
        # U1 = Conv3x3(padding=1), U2 = Conv3x3(dilation=2, padding=2)
        # Both branches output in_ch channels; concatenation yields 2*in_ch
        self.multi_scale = nn.ModuleList([
            nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1, bias=False),
            nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=2, dilation=2, bias=False)
        ])
        self.bn = nn.BatchNorm2d(in_ch * 2)
        self.relu = nn.ReLU(inplace=True)

        # SE-style channel attention with reduction ratio gamma=8
        se_mid_ch = (in_ch * 2) // gamma
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch * 2, se_mid_ch, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(se_mid_ch, in_ch * 2, 1, bias=False),
            nn.Sigmoid()
        )

        # Channel calibration and residual
        self.calibrate = nn.Conv2d(in_ch * 2, out_ch, kernel_size=1, bias=False)
        self.residual = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False) if in_ch != out_ch else nn.Identity()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        # 1. Dual-branch multi-scale feature extraction + concat + BN + ReLU
        scale_feats = torch.cat([conv(x) for conv in self.multi_scale], dim=1)
        scale_feats = self.relu(self.bn(scale_feats))

        # 2. Channel attention weighting
        attn_weight = self.channel_attention(scale_feats)
        attended_feats = scale_feats * attn_weight

        # 3. Channel calibration + residual connection + DropPath
        calibrated = self.calibrate(attended_feats)
        return self.residual(x) + self.drop_path(calibrated)


# --------------------------- Main Network: HiFuse-SGKAN ---------------------------
class HiFuse_SGKAN(nn.Module):
    """Medical image segmentation network: encoder + SGKAN hybrid attention backbone + progressive decoder."""

    def __init__(self, num_classes, input_channels=3, deep_supervision=False, img_size=224,
                 patch_size=16, in_chans=3, embed_dims=[256, 320, 512], no_kan=False,
                 drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm, depths=[1, 1, 1],
                 global_dims=[8, 16, 32, 64, 128], **kwargs):
        super().__init__()
        embed_dims = [128, 160, 256]
        kan_input_dim = embed_dims[0]

        # Encoder
        self.encoder1 = ConvLayer(3, kan_input_dim // 8)
        self.encoder2 = ConvLayer(kan_input_dim // 8, kan_input_dim // 4)
        self.encoder3 = ConvLayer(kan_input_dim // 4, kan_input_dim // 2)
        self.encoder4 = ConvLayer(kan_input_dim // 2, kan_input_dim)

        # SG-KAN hybrid attention backbone
        self.backbone = nn.ModuleList([
            SGKAN_Hybrid(kan_input_dim, kan_input_dim, **kwargs) for _ in range(3)
        ])

        # Decoder
        self.decoder2 = D_ConvLayer(kan_input_dim, kan_input_dim // 2)
        self.decoder3 = D_ConvLayer(kan_input_dim // 2, kan_input_dim // 4)
        self.decoder4 = D_ConvLayer(kan_input_dim // 4, kan_input_dim // 8)
        self.decoder5 = D_ConvLayer(kan_input_dim // 8, kan_input_dim // 8)

        # Output layer
        self.final = nn.Conv2d(kan_input_dim // 8, num_classes, kernel_size=1)

        # Enhancement modules
        self.sbe1 = SBE(in_ch=kan_input_dim // 8, out_ch=kan_input_dim // 8, drop_path=drop_path_rate * 0.3)
        self.sbe2 = SBE(in_ch=kan_input_dim // 4, out_ch=kan_input_dim // 4, drop_path=drop_path_rate * 0.3)
        self.dca3 = DCA(in_ch=kan_input_dim // 2, out_ch=kan_input_dim // 2, drop_path=drop_path_rate)

    def forward(self, x):
        # Encoder downsampling
        out = F.relu(F.max_pool2d(self.encoder1(x), 2, 2))
        t1 = self.sbe1(out)

        out = F.relu(F.max_pool2d(self.encoder2(out), 2, 2))
        t2 = self.sbe2(out)

        out = F.relu(F.max_pool2d(self.encoder3(out), 2, 2))
        t3 = self.dca3(out)

        out = F.relu(F.max_pool2d(self.encoder4(out), 2, 2))

        # SGKAN feature enhancement
        for layer in self.backbone:
            out = layer(out)

        # Progressive upsampling decoder with skip connections
        out = F.relu(F.interpolate(self.decoder2(out), scale_factor=2, mode='bilinear'))
        out = torch.add(out, t3)

        out = F.relu(F.interpolate(self.decoder3(out), scale_factor=2, mode='bilinear'))
        out = torch.add(out, t2)

        out = F.relu(F.interpolate(self.decoder4(out), scale_factor=2, mode='bilinear'))
        out = torch.add(out, t1)

        out = F.relu(F.interpolate(self.decoder5(out), scale_factor=2, mode='bilinear'))

        return self.final(out)
