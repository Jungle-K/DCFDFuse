import torch
import numbers
import torch.nn as nn
from einops import rearrange
import torch.nn.functional as F


"""
input:ir_low,vis_low    [batchsize, 1, H, W]
output:low_fusion       [batchsize, 3, H, W]
"""

##########################################################################
## ViTransformer
class AttentionBase(nn.Module):
    def __init__(self,
                 dim,
                 num_heads=8,
                 qkv_bias=False, ):
        super(AttentionBase, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv1 = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=qkv_bias)
        self.qkv2 = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, padding=1, bias=qkv_bias)
        self.proj = nn.Conv2d(dim, dim, kernel_size=1, bias=qkv_bias)

    def forward(self, x):
        # [batch_size, num_patches + 1, total_embed_dim]
        b, c, h, w = x.shape
        qkv = self.qkv2(self.qkv1(x))
        q, k, v = qkv.chunk(3, dim=1)
        q = rearrange(q, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        # transpose: -> [batch_size, num_heads, embed_dim_per_head, num_patches + 1]
        # @: multiply -> [batch_size, num_heads, num_patches + 1, num_patches + 1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w',
                        head=self.num_heads, h=h, w=w)

        out = self.proj(out)
        return out


class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(
            dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w',
                        head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out


class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(
            dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3,
                                stride=1, padding=1, groups=hidden_features*2, bias=bias)

        self.project_out = nn.Conv2d(
            hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


class Mlp(nn.Module):
    """
    MLP as used in Vision Transformer, MLP-Mixer and related networks
    """

    def __init__(self,
                 in_features,
                 hidden_features=None,
                 ffn_expansion_factor=2,
                 bias=False):
        super().__init__()
        hidden_features = int(in_features * ffn_expansion_factor)

        self.project_in = nn.Conv2d(
            in_features, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3,
                                stride=1, padding=1, groups=hidden_features, bias=bias)

        self.project_out = nn.Conv2d(
            hidden_features, in_features, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class BaseFeatureExtraction(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 ffn_expansion_factor=1.,
                 qkv_bias=False, ):
        super(BaseFeatureExtraction, self).__init__()
        self.norm1 = LayerNorm(dim, 'WithBias')
        self.attn = AttentionBase(dim, num_heads=num_heads, qkv_bias=qkv_bias, )
        self.norm2 = LayerNorm(dim, 'WithBias')
        self.mlp = Mlp(in_features=dim,
                       ffn_expansion_factor=ffn_expansion_factor, )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


# import timm
# class ViTFeatureExtractor(nn.Module):
#     def __init__(self, layer_num=6):
#         super(ViTFeatureExtractor, self).__init__()
#         self.layer_num = layer_num
#         self.model = timm.create_model('vit_base_patch16_224', pretrained=True, in_chans=1)
#
#     def forward(self, x):
#         intermediate_features = self.model.get_intermediate_layers(x, self.layer_num)
#         return intermediate_features

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


class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)
        return x


class FusionNetwork_l(nn.Module):
    def __init__(self, trans_channels, fusion_channels, out_channels):
        super(FusionNetwork_l, self).__init__()
        self.vit = BaseFeatureExtraction(dim=trans_channels, num_heads=8)
        self.single_conv = Single_Conv(trans_channels, trans_channels, activate=True, bn=True)
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(2 * trans_channels, fusion_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(fusion_channels, fusion_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(fusion_channels),
            nn.ReLU(inplace=True)
        )
        self.channel_attention = ChannelAttention(in_channels=fusion_channels + 2 * trans_channels)
        self.output_conv = nn.Conv2d(fusion_channels + 2 * trans_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, ir_low, vis_low):
        # ir_low, vis_low
        # ir_low = img
        # vis_low = img
        ir_low_trans = self.vit(ir_low)
        vis_low_trans = self.vit(vis_low)
        ir_low_features = self.single_conv(ir_low_trans)
        vis_low_features = self.single_conv(vis_low_trans)

        combined = torch.cat((ir_low_features, vis_low_features), dim=1)
        features = self.fusion_conv(combined)
        fusion_combined = torch.cat((features, ir_low_trans, vis_low_trans), dim=1)
        attention = self.channel_attention(fusion_combined)
        output = self.output_conv(attention)

        return output


class fusion_low(nn.Module):
    def __init__(self):
        super(fusion_low, self).__init__()
        # v3.8
        # self.overlap = OverlapPatchEmbed(in_c=1, embed_dim=32)
        # self.transformer = BaseFeatureExtraction(dim=32, num_heads=8)
        # v3.9
        self.conv = Single_Conv(1, 32, kernel_size=3, padding=1, activate=True, bn=True)
        self.conv_down = Single_Conv(64, 128, kernel_size=3, padding=1, activate=True, bn=True)
        self.conv_down_2 = Single_Conv(128, 256, kernel_size=3, padding=1, activate=True, bn=True)
        self.down = nn.MaxPool2d(kernel_size=2)
        self.conv_up_2 = Single_Conv(256, 128, kernel_size=3, padding=1, activate=True, bn=True)
        self.conv_up = Single_Conv(256, 64, kernel_size=3, padding=1, activate=True, bn=True)
        # v3.5
        # self.conv_up = Single_Conv(128, 64, kernel_size=3, padding=1, activate=True, bn=True)

        self.conv_transpose_2 = torch.nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)
        self.conv_transpose_1 = torch.nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)

    def forward(self, ir_low, vis_low):
        # ir_layer = self.transformer(self.overlap(ir_low))
        # vis_layer = self.transformer(self.overlap(vis_low))
        ir_layer = self.conv(ir_low)
        vis_layer = self.conv(vis_low)

        low_fusion_1 = self.conv_down(torch.cat((ir_layer, vis_layer), dim=1))  # [batchsize, 128, H, W]
        low_fusion_2 = self.down(low_fusion_1)  # [batchsize, 128, H/2, W/2]
        low_fusion_3 = self.conv_down_2(low_fusion_2)  # [batchsize, 256, H/2, W/2]
        low_fusion_4 = self.down(low_fusion_3)  # [batchsize, 256, H/4, W/4]

        low_fusion_5 = self.conv_up_2(low_fusion_4)  # [batchsize, 128, H/4, W/4]
        low_fusion_6 = self.conv_transpose_2(low_fusion_5)  # [batchsize, 128, H/2, W/2]
        low_fusion_7 = self.conv_up(torch.cat((low_fusion_6, low_fusion_2), dim=1))  # [batchsize, 64, H/2, W/2]
        low_fusion = self.conv_transpose_1(low_fusion_7)  # [batchsize, 64, H, W]

        # v3.5
        # low_fusion_1 = self.conv_down(torch.cat((ir_layer, vis_layer), dim=1))  # [batchsize, 128, H, W]
        # low_fusion_2 = self.conv_down_2(self.down(low_fusion_1))  # [batchsize, 256, H/2, W/2]
        # low_fusion_3 = self.conv_up_2(self.down(low_fusion_2))  # [batchsize, 128, H/4, W/4]
        # low_fusion_4 = self.conv_up(self.conv_transpose_2(low_fusion_3))  # [batchsize, 64, H/2, W/2]
        # low_fusion = self.conv_transpose_1(low_fusion_4)  # [batchsize, 64, H, W]

        return low_fusion


class low_loss2(nn.Module):
    def __init__(self):
        super(low_loss2, self).__init__()
        self.start_conv = Single_Conv(1, 32, kernel_size=3, padding=1, activate=True, bn=True) # cat32/add64
        # self.transformer = nn.Sequential(*[TransformerBlock(dim=64, num_heads=8, ffn_expansion_factor=2,
        #                                     bias=False, LayerNorm_type='WithBias') for i in range(3)])
        self.basefeature = BaseFeatureExtraction(dim=64, num_heads=8)
        self.add_conv = Single_Conv(128, 64, activate=True, bn=True)
        self.out_conv = Single_Conv(64, 64, activate=True,bn=True)
        self.drop = nn.Dropout(p=0.2)


    def forward(self, ir_img, vis_img, ir_low, vis_low):
        # # add
        # ir_img = self.start_conv(ir_img)
        # vis_img = self.start_conv(vis_img)
        # ir_low = self.start_conv(ir_low)
        # vis_low = self.start_conv(vis_low)
        # ir_layer = self.transformer(ir_img + ir_low)
        # vis_layer = self.transformer(vis_img + vis_low)
        # low_layer = self.basefeature(torch.cat((ir_layer, vis_layer), dim=1))
        # output = self.end_conv(low_layer)
        # cat
        ir_img = self.start_conv(ir_img)
        vis_img = self.start_conv(vis_img)
        ir_low = self.start_conv(ir_low)
        vis_low = self.start_conv(vis_low)
        ir_low_layer = self.basefeature(torch.cat((ir_img, ir_low), dim=1))
        vis_low_layer = self.basefeature(torch.cat((vis_img, vis_low), dim=1))
        low_layer = self.add_conv(torch.cat((ir_low_layer, vis_low_layer), dim=1))
        output = self.out_conv(low_layer)
        # output = self.drop(output)
        return output, ir_low_layer, vis_low_layer


def low_flops():
    from ptflops import get_model_complexity_info

    fusion_model = FusionNetwork_l(trans_channels=16, fusion_channels=32, out_channels=3)
    ir_low = torch.randn(4, 1, 32, 32)
    vis_low = torch.randn(4, 1, 32, 32)
    fused_image = fusion_model(ir_low, vis_low)
    flops, params = get_model_complexity_info(fusion_model, (1, 224, 224), as_strings=True, print_per_layer_stat=True)
    print('flops: ', flops, 'params: ', params)


def debug_transformer():
    input = torch.randn(4, 1, 32, 32)
    transformer_model = BaseFeatureExtraction(dim=16, num_heads=8)
    output = transformer_model(input)
    print(output.shape)


def debug_low():
    fusion_model = FusionNetwork_l(trans_channels=16, fusion_channels=32, out_channels=3)
    model = low_loss2()

    ir_low = torch.randn(4, 1, 32, 32)
    vis_low = torch.randn(4, 1, 32, 32)

    fused_image,_,_ = model(ir_low, ir_low, ir_low, ir_low)

    print(fused_image.shape)  # 输出处理后的张量的形状


if __name__ == '__main__':
    debug_low()
