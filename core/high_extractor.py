import torch
import torch.nn as nn
from core.low_extractor import TransformerBlock


"""
input:ir_high,vis_high    [batchsize, 1, H, W]
output:high_fusion       [batchsize, 3, H, W]
"""

##########################################################################
## Channel Attention
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return x * self.sigmoid(out)


##########################################################################
## Feature extractor
class Single_Conv(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 padding=1,
                 activate=True,
                 bn=False):
        super(Single_Conv, self).__init__()
        self.activate = activate
        self.bn = bn
        self.conv = nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              padding=padding)
        self.relu = nn.ReLU(inplace=True)
        self.batch_norm = nn.BatchNorm2d(out_channels)

    def forward(self, image):
        out = self.conv(image)
        if self.bn:
            out = self.batch_norm(out)
        if self.activate:
            out = self.relu(out)
        return out


class Trans_cnn(nn.Module):
    def __init__(self, out_channels, trans_channels=16, num_heads=8):
        super(Trans_cnn, self).__init__()
        self.vit = TransformerBlock(dim=trans_channels, num_heads=num_heads,
                                    ffn_expansion_factor=2, bias=False, LayerNorm_type='WithBias')
        self.after_conv = Single_Conv(trans_channels, out_channels, activate=True, bn=True)
    def forward(self, x):
        x = self.after_conv(self.vit(x))
        return x


##########################################################################
## INN
class InvertedResidualBlock(nn.Module):
    def __init__(self, inp, oup, expand_ratio):
        super(InvertedResidualBlock, self).__init__()
        hidden_dim = int(inp * expand_ratio)
        self.bottleneckBlock = nn.Sequential(
            # pw
            nn.Conv2d(inp, hidden_dim, 1, bias=False),
            # nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            # dw
            nn.ReflectionPad2d(1),
            nn.Conv2d(hidden_dim, hidden_dim, 3, groups=hidden_dim, bias=False),
            # nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, bias=False),
            # nn.BatchNorm2d(oup),
        )
    def forward(self, x):
        return self.bottleneckBlock(x)

class DetailNode(nn.Module):
    def __init__(self):
        super(DetailNode, self).__init__()
        # Scale is Ax + b, i.e. affine transformation
        self.theta_phi = InvertedResidualBlock(inp=32, oup=32, expand_ratio=2)
        self.theta_rho = InvertedResidualBlock(inp=32, oup=32, expand_ratio=2)
        self.theta_eta = InvertedResidualBlock(inp=32, oup=32, expand_ratio=2)
        self.shffleconv = nn.Conv2d(64, 64, kernel_size=1,
                                    stride=1, padding=0, bias=True)
    def separateFeature(self, x):
        z1, z2 = x[:, :x.shape[1]//2], x[:, x.shape[1]//2:x.shape[1]]
        return z1, z2
    def forward(self, z1, z2):
        z1, z2 = self.separateFeature(
            self.shffleconv(torch.cat((z1, z2), dim=1)))
        z2 = z2 + self.theta_phi(z1)
        z1 = z1 * torch.exp(self.theta_rho(z2)) + self.theta_eta(z2)
        return z1, z2

# INN
class DetailFeatureExtraction(nn.Module):
    def __init__(self, num_layers=3):
        super(DetailFeatureExtraction, self).__init__()
        INNmodules = [DetailNode() for _ in range(num_layers)]
        self.net = nn.Sequential(*INNmodules)
    def forward(self, x):
        z1, z2 = x[:, :x.shape[1]//2], x[:, x.shape[1]//2:x.shape[1]]
        for layer in self.net:
            z1, z2 = layer(z1, z2)
        return torch.cat((z1, z2), dim=1)


##########################################################################
# network

class FusionNetwork_h(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FusionNetwork_h, self).__init__()
        # self.overlappatch = nn.Conv2d(in_channels, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_1d = Single_Conv(in_channels, 2 * in_channels, activate=True, bn=True)
        self.conv_2d = Single_Conv(2 * in_channels, 4 * in_channels, activate=True, bn=True)
        self.conv_3d = Single_Conv(4 * in_channels, 8 * in_channels, activate=True, bn=True)
        self.down = nn.MaxPool2d(kernel_size=2)
        self.conv_3u = Single_Conv(16 * in_channels, 4 * in_channels, activate=True, bn=True)
        self.conv_2u = Single_Conv(12 * in_channels, 2 * in_channels, activate=True, bn=True)
        self.conv_1u = Single_Conv(6 * in_channels, out_channels, activate=True, bn=True)
        self.channel_attention = ChannelAttention(in_channels=out_channels)

    def forward(self, ir_high, vis_high):
        # ir_high, vis_high
        # ir_high= img
        # vis_high = img
        ir_high2 = self.conv_1d(ir_high)
        ir_high_2down = self.down(ir_high2)
        vis_high2 = self.conv_1d(vis_high)
        vis_high_2down = self.down(vis_high2)

        ir_high4 = self.conv_2d(ir_high_2down)
        ir_high_4down = self.down(ir_high4)
        vis_high4 = self.conv_2d(vis_high_2down)
        vis_high_4down = self.down(vis_high4)

        ir_high8 = self.conv_3d(ir_high_4down)
        # ir_high_8down = self.down(ir_high8)
        vis_high8 = self.conv_3d(vis_high_4down)
        # vis_high_8down = self.down(vis_high8)

        combined_3 = self.conv_3u(torch.cat((ir_high8, vis_high8), dim=1))
        comb_2in = nn.functional.interpolate(combined_3, scale_factor=2, mode='bilinear', align_corners=False)
        combined_2 = self.conv_2u(torch.cat((comb_2in, ir_high4, vis_high4), dim=1))
        comb_1in = nn.functional.interpolate(combined_2, scale_factor=2, mode='bilinear', align_corners=False)
        output = self.conv_1u(torch.cat((comb_1in, ir_high2, vis_high2), dim=1))
        # output = nn.functional.interpolate(combined_1, scale_factor=2, mode='bilinear', align_corners=False)

        return output


class fusion_high(nn.Module):
    def __init__(self):
        super(fusion_high, self).__init__()
        self.conv = Single_Conv(1, 64, activate=True, bn=True)
        self.inn = DetailFeatureExtraction()
        self.add_conv = Single_Conv(128, 64, kernel_size=3, padding=1, activate=True, bn=True)
        self.add_conv_2 = Single_Conv(64, 64, kernel_size=3, padding=1, activate=True, bn=True)

    def forward(self, ir_high, vis_high):
        ir_layer = self.inn(self.conv(ir_high))
        vis_layer = self.inn(self.conv(vis_high))
        output = self.add_conv(torch.cat((ir_layer, vis_layer), dim=1))
        output = self.add_conv_2(output)
        return output


class high_loss2(nn.Module):
    def __init__(self):
        super(high_loss2, self).__init__()
        self.start_conv = Single_Conv(1, 32, activate=True, bn=True)
        self.inn = DetailFeatureExtraction()
        self.add_conv = Single_Conv(128, 64, kernel_size=3, padding=1, activate=True, bn=True)
        self.out_conv = Single_Conv(64, 64, kernel_size=3, padding=1, activate=True, bn=True)
        # self.drop = nn.Dropout(p=0.2)

    def forward(self, ir_img, vis_img, ir_high, vis_high):
        ir_img = self.start_conv(ir_img)
        vis_img = self.start_conv(vis_img)
        ir_high = self.start_conv(ir_high)
        vis_high = self.start_conv(vis_high)
        ir_high_layer = self.inn(torch.cat((ir_img, ir_high), dim=1))
        vis_high_layer = self.inn(torch.cat((vis_img, vis_high), dim=1))
        high_layer = self.add_conv(torch.cat((ir_high_layer, vis_high_layer), dim=1))
        output = self.out_conv(high_layer)
        # output = self.drop(output)
        return output, ir_high_layer, vis_high_layer


def high_flops():
    from ptflops import get_model_complexity_info

    fusion_model = FusionNetwork_h(in_channels=1, out_channels=3)
    img = torch.randn(4, 1, 32, 32)
    fused_image = fusion_model(img)
    flops, params = get_model_complexity_info(fusion_model, (1, 224, 224), as_strings=True, print_per_layer_stat=True)
    print('flops: ', flops, 'params: ', params)


def debug_high():
    # 创建一个BaseFeatureExtraction实例
    fusion_model = FusionNetwork_h(in_channels=1, out_channels=3)
    # 生成一个输入张量
    ir_high = torch.randn(4, 1, 32, 32)
    vis_high = torch.randn(4, 1, 32, 32)
    # 将输入张量传入BaseFeatureExtraction方法
    fused_image = fusion_model(ir_high, vis_high)

    print(fused_image.shape)  # 输出处理后的张量的形状

def debug_high_2():
    model = high_loss2()
    ir_input = torch.randn(4, 1, 32, 32)
    vis_input = torch.randn(4, 1, 32, 32)
    out,_,_ = model(ir_input, vis_input, ir_input, vis_input)
    print(out.shape)

if __name__ == '__main__':
    debug_high_2()