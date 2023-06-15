import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.config import cfg

class AppNet(nn.Module):
  def __init__(self, input_dim_a=3):
    super(AppNet, self).__init__()
    output_nc = cfg.dim_app_emb
    dim = 64
    self.model = nn.Sequential(
        nn.ReflectionPad2d(3),
        nn.Conv2d(input_dim_a, dim, 7, 1),
        nn.ReLU(inplace=True),  ## size
        nn.ReflectionPad2d(1),
        nn.Conv2d(dim, dim*2, 4, 2),
        nn.ReLU(inplace=True),  ## size/2
        nn.ReflectionPad2d(1),
        nn.Conv2d(dim*2, dim*4, 4, 2),
        nn.ReLU(inplace=True),  ## size/4
        nn.ReflectionPad2d(1),
        nn.Conv2d(dim*4, dim*4, 4, 2),
        nn.ReLU(inplace=True),  ## size/8
        nn.ReflectionPad2d(1),
        nn.Conv2d(dim*4, dim*4, 4, 2),
        nn.ReLU(inplace=True),  ## size/16
        nn.AdaptiveAvgPool2d(1),
        nn.Conv2d(dim*4, output_nc, 1, 1, 0))  ## 1*1

  def forward(self, x):
    x = self.model(x)
    output = x.view(x.size(0), -1)
    return output

class MaskNet(nn.Module):
    def __init__(self, W=256, input_ch_uv=42):
        super(MaskNet, self).__init__()
        self.embeddings = nn.Embedding(cfg.n_emb, cfg.dim_mask_emb)
        self.mask_mapping = nn.Sequential(
                            nn.Linear(cfg.dim_mask_emb + input_ch_uv, W), nn.ReLU(True),
                            nn.Linear(W, W), nn.ReLU(True),
                            nn.Linear(W, W), nn.ReLU(True),
                            nn.Linear(W, W), nn.ReLU(True),
                            nn.Linear(W, 1), nn.Sigmoid())

    def forward(self, uv, emb_id):
        emb = self.embeddings(emb_id)
        x = torch.cat([uv, emb], dim=-1)
        mask = self.mask_mapping(x)
        return mask

class EnvNet(nn.Module):
    def __init__(self, W=256, input_ch_uv=2):
        super(EnvNet, self).__init__()
        self.embeddings = nn.Embedding(cfg.n_emb, cfg.dim_env_emb)
        self.net = nn.Sequential(
                            nn.Linear(input_ch_uv + cfg.dim_env_emb, W), nn.ReLU(True),
                            nn.Linear(W, W), nn.ReLU(True),
                            nn.Linear(W, W), nn.ReLU(True),
                            nn.Linear(W, W), nn.ReLU(True),
                            nn.Linear(W, 3), nn.Sigmoid())

    def forward(self, x, emb_id, batch=None):
        emb = self.embeddings(emb_id)
        x = torch.cat([x, emb], dim=-1)
        return self.net(x)

class MaskConvNet(nn.Module):
    def __init__(self, W=256, input_ch_uv=42):
        super(MaskConvNet, self).__init__()
        self.feature_net = FeatureNet()

    def forward(self, uv, inp):
        mask = self.feature_net(inp)
        mask = F.grid_sample(mask, uv[None, None]*2-1., align_corners=False, mode='bilinear', padding_mode='border')[0].permute(1, 2, 0)[0]
        return torch.sigmoid(mask)


class ConvBnReLU(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size=3, stride=1, pad=1,
                 norm_act=nn.BatchNorm2d):
        super(ConvBnReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size, stride=stride, padding=pad, bias=False)
        self.bn = norm_act(out_channels)
        self.relu = nn.ReLU(inplace=True)


    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class FeatureNet(nn.Module):
    def __init__(self, norm_act=nn.BatchNorm2d):
        super(FeatureNet, self).__init__()
        self.conv0 = nn.Sequential(
                        ConvBnReLU(3, 8, 3, 1, 1, norm_act=norm_act),
                        ConvBnReLU(8, 8, 3, 1, 1, norm_act=norm_act))
        self.conv1 = nn.Sequential(
                        ConvBnReLU(8, 16, 5, 2, 2, norm_act=norm_act),
                        ConvBnReLU(16, 16, 3, 1, 1, norm_act=norm_act))
        self.conv2 = nn.Sequential(
                        ConvBnReLU(16, 32, 5, 2, 2, norm_act=norm_act),
                        ConvBnReLU(32, 32, 3, 1, 1, norm_act=norm_act))
        self.toplayer = nn.Conv2d(32, 32, 1)
        self.lat1 = nn.Conv2d(16, 32, 1)
        self.lat0 = nn.Conv2d(8, 32, 1)
        self.smooth1 = nn.Conv2d(32, 16, 3, padding=1)
        self.smooth0 = nn.Conv2d(32, 1, 3, padding=1)

    def _upsample_add(self, x, y):
        return F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True) + y

    def forward(self, x):
        conv0 = self.conv0(x)
        conv1 = self.conv1(conv0)
        conv2 = self.conv2(conv1)
        feat2 = self.toplayer(conv2)
        feat1 = self._upsample_add(feat2, self.lat1(conv1))
        feat0 = self._upsample_add(feat1, self.lat0(conv0))
        feat1 = self.smooth1(feat1)
        feat0 = self.smooth0(feat0)
        return feat0
