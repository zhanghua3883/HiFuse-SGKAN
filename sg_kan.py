import torch
import torch.nn as nn
import torch.nn.functional as F

# Only keep conv3x3 as required by the codebase
def conv3x3(in_ch, out_ch, **kwargs):
    return nn.Conv2d(in_ch, out_ch, 3, padding=1, **kwargs)

# Only keep _make_layer as required by the codebase (logic unchanged)
def _make_layer(layer, *args, **kwargs):
    if layer == "conv3x3":
        return conv3x3(*args, **kwargs)
    else:
        raise ValueError(f"Unrecognized `layer` found: {layer}.")

# Original ResBlock preserved (logic fully unchanged)
class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch=None, hid_ch=None, bn=False, layer="conv3x3", **kwargs):
        super(ResBlock, self).__init__()

        if out_ch is None:
            out_ch = in_ch
        if hid_ch is None:
            hid_ch = min(in_ch, out_ch)

        self.block1 = nn.Sequential(
            nn.BatchNorm2d(in_ch) if bn else nn.Identity(),
            nn.ReLU(),
            _make_layer(layer, in_ch, hid_ch, **kwargs)
        )

        self.block2 = nn.Sequential(
            nn.BatchNorm2d(hid_ch) if bn else nn.Identity(),
            nn.ReLU(),
            _make_layer(layer, hid_ch, out_ch, **kwargs)
        )

        if in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.BatchNorm2d(in_ch) if bn else nn.Identity(),
                _make_layer(layer, in_ch, out_ch, bias=False, **kwargs)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = x
        x = self.block1(x)
        x = self.block2(x)
        return self.shortcut(identity) + x

# Polynomial basis functions
class Polynomial2d(nn.Module):
    def __init__(self, in_channels, r=6):
        super(Polynomial2d, self).__init__()
        self.r = r
        self.offset = ResBlock(in_channels, in_channels*2, bn=True)

    def forward(self, x):
        B, n, H, W = x.shape

        x_grid, y_grid = torch.meshgrid(torch.linspace(-1, 1, H, device=x.device),
                                        torch.linspace(-1, 1, W, device=x.device),
                                        indexing='xy') #torch.Size([16, 16])

        x_delta, y_delta = torch.split(self.offset(x), n, dim=1)#torch.Size([8, 128, 16, 16])

        x_grid = x_delta + x_grid.unsqueeze(0).unsqueeze(0) #torch.Size([8, 128, 16, 16])
        y_grid = y_delta + y_grid.unsqueeze(0).unsqueeze(0)
        
        basis_functions = []
        idx = 0
        degree = 0
        while len(basis_functions) < self.r:
            for i in range(degree + 1):
                if len(basis_functions) >= self.r:
                    break
                j = degree - i
                basis_functions.append(x_grid**i * y_grid**j)
            degree += 1
        
        return torch.stack(basis_functions, 2)

# Base
class SGKAN_Base(nn.Module):
    def __init__(self, in_channels, out_channels, r=6, **kwargs):
        super(SGKAN_Base, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.r = r
        self.psi = Polynomial2d(in_channels, r=r)
        self.theta = nn.Conv2d(in_channels, out_channels, 1)
        
    def compute_basis_responses(self, x):
        psi = self.psi(x)
        return psi

# Main module: SGKAN_Hybrid
class SGKAN_Hybrid(SGKAN_Base):
    def __init__(self, in_channels, out_channels, r=6, reduction=4, **kwargs):
        super(SGKAN_Hybrid, self).__init__(in_channels, out_channels, r,** kwargs)
        self.rho = reduction

        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid()
        )

        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // self.rho, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // self.rho, in_channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        else:
            self.shortcut = nn.Identity()
                # Added: attention logits over the basis dimension, enabling the attention to weight different polynomial bases
        # self.basis_attention = nn.Conv2d(in_channels, r, kernel_size=1, bias=False)

    def forward(self, x):
        B, C, H, W = x.shape
        basis_responses = self.compute_basis_responses(x)

        spatial_avg = torch.mean(x, dim=1, keepdim=True)
        spatial_max = torch.max(x, dim=1, keepdim=True)[0]
        spatial_feat = torch.cat([spatial_avg, spatial_max], dim=1)
        
        spatial_attn_map = self.spatial_attention(spatial_feat).unsqueeze(1)
        channel_attn_vec = self.channel_attention(x).unsqueeze(2)

        joint_attn_weight = spatial_attn_map * channel_attn_vec
        # basis_logits = self.basis_attention(x).unsqueeze(1)
        # joint_attn_weight = spatial_attn_map * channel_attn_vec * basis_logits
        # attn_score = F.softmax(joint_attn_weight, dim=2)
        refined_features = (basis_responses * joint_attn_weight).sum(dim=2)

        out = self.theta(refined_features) + self.shortcut(x)
        return out
