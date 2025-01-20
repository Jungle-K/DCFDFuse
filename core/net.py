import torch
import torch.nn as nn
from core.block import OverlapPatchEmbed, DetailFeatureExtraction
from core.block import BaseFeatureExtraction, TransformerBlock
from core.block import Single_Conv, ConvLayer, FusionBlock_res, DenseBlock_light, DCB, UpsampleReshape_eval
from core.FSFusion_strategy import attention_fusion_weight


class high_extractor(nn.Module):
    def __init__(self):
        super(high_extractor, self).__init__()
        self.op_ir = OverlapPatchEmbed(in_c=1, embed_dim=64)
        self.op_vis = OverlapPatchEmbed(in_c=1, embed_dim=64)
        self.INN_ir = DetailFeatureExtraction(num_layers=2)
        self.INN_vis = DetailFeatureExtraction(num_layers=2)
        self.sigmoid = nn.Sigmoid()

        self.conv_1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(32),
            # nn.ReLU(inplace=True),
            # nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        )
        self.conv_2 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(32),
            # nn.ReLU(inplace=True),
            # nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        )

    def forward(self, ir_img, vis_img, ir_high=None, vis_high=None):
        # ir_img, vis_img, ir_high, vis_high = img, img, img, img
        # ir_img = self.op_ir(ir_img)
        # vis_img = self.op_vis(vis_img)  # C(1->64)
        ir_inn = self.INN_ir(ir_img)
        vis_inn = self.INN_vis(vis_img)

        return ir_inn, vis_inn


class low_extractor(nn.Module):
    def __init__(self):
        super(low_extractor, self).__init__()
        self.op_ir = OverlapPatchEmbed(in_c=1, embed_dim=64)
        self.op_vis = OverlapPatchEmbed(in_c=1, embed_dim=64)
        self.BASE_ir = BaseFeatureExtraction(dim=64, num_heads=8)
        self.BASE_vis = BaseFeatureExtraction(dim=64, num_heads=8)
        self.sigmoid = nn.Sigmoid()

        self.conv_1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False)
        )
        self.conv_2 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False)
        )

    def forward(self, ir_img, vis_img, ir_low=None, vis_low=None):
        # ir_img, vis_img, ir_high, vis_high = img, img, img, img
        # ir_img = self.op_ir(ir_img)
        # vis_img = self.op_vis(vis_img)  # C(1->64)
        ir_base = self.BASE_ir(ir_img)
        vis_base = self.BASE_vis(vis_img)
        return ir_base, vis_base  # N, 64, H, W


class trans_encoder(nn.Module):
    def __init__(self,
                 inp_channels=1,
                 out_channels=1,
                 dim=64,
                 num_blocks=4,
                 heads=8,
                 ffn_expansion_factor=2,
                 bias=False,
                 LayerNorm_type='WithBias',
                 ):
        super(trans_encoder, self).__init__()

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        self.encoder_level1 = nn.Sequential(
            *[TransformerBlock(dim=dim, num_heads=heads, ffn_expansion_factor=ffn_expansion_factor,
                               bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks)])
        self.baseFeature = BaseFeatureExtraction(dim=dim, num_heads=heads)
        self.detailFeature = DetailFeatureExtraction()

    def forward(self, inp_img):
        inp_enc_level1 = self.patch_embed(inp_img)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)
        base_feature = self.baseFeature(out_enc_level1)
        detail_feature = self.detailFeature(out_enc_level1)
        return base_feature, detail_feature  # N, 64, H, W


class cnn_encoder(nn.Module):
    def __init__(self):
        super(cnn_encoder, self).__init__()
        kernel_size = 3
        stride = 1

        self.conv1_ir = ConvLayer(1, 16, kernel_size, stride)
        self.DB_ir = DenseBlock_light(16, kernel_size, stride)
        self.conv1_vis = ConvLayer(1, 16, kernel_size, stride)
        self.DB_vis = DenseBlock_light(16, kernel_size, stride)

    def forward(self, ir_img, vis_img):
        ir_cnn = self.DB_ir(self.conv1_ir(ir_img))
        vis_cnn = self.DB_vis(self.conv1_vis(vis_img))
        return ir_cnn, vis_cnn  # N, 64, H, W

class fuse_1(nn.Module):
    def __init__(self):
        super(fuse_1, self).__init__()
        self.sigmoid = nn.Sigmoid()

        kernel_size = 3
        stride = 1

        self.base_rfn = FusionBlock_res(64)
        self.inn_rfn = FusionBlock_res(64)
        # decoder
        # self.conv2 = ConvLayer(128, 128, kernel_size, stride)
        self.conv3 = ConvLayer(64, 64, kernel_size, stride)
        self.conv4 = ConvLayer(64, 32, kernel_size, stride)
        self.conv5 = ConvLayer(32, 1, kernel_size, stride, is_last=True)

    def forward(self, vis, ir_base, vis_base, ir_detail=None, vis_detail=None):
        if ir_detail != None:
            base_layer, base_init = self.base_rfn(ir_base, vis_base)
            inn_layer, inn_init = self.inn_rfn(ir_detail, vis_detail)
            base_layer = base_layer + inn_init
            inn_layer = inn_layer + base_init
        else:
            base_layer = ir_base
            inn_layer = vis_base  # N, 64, H, W

        # # 消融实验 add
        # base_layer = (ir_base + vis_base)/2
        # inn_layer = (ir_detail + vis_detail)/2

        fuse_layer = attention_fusion_weight(base_layer, inn_layer)
        output = self.conv5(self.conv4(self.conv3(fuse_layer)))
        output = output + vis
        return self.sigmoid(output)
        # return output


#######################################################################################
# cddfuse
class Restormer_Encoder(nn.Module):
    def __init__(self,
                 inp_channels=1,
                 out_channels=1,
                 dim=64,
                 num_blocks=[4, 4],
                 heads=[8, 8, 8],
                 ffn_expansion_factor=2,
                 bias=False,
                 LayerNorm_type='WithBias',
                 ):
        super(Restormer_Encoder, self).__init__()

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        self.encoder_level1 = nn.Sequential(
            *[TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                               bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        self.baseFeature = BaseFeatureExtraction(dim=dim, num_heads=heads[2])
        self.detailFeature = DetailFeatureExtraction()

    def forward(self, inp_img):
        inp_enc_level1 = self.patch_embed(inp_img)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)
        base_feature = self.baseFeature(out_enc_level1)
        detail_feature = self.detailFeature(out_enc_level1)
        return base_feature, detail_feature, out_enc_level1
class Restormer_Decoder(nn.Module):
    def __init__(self,
                 inp_channels=1,
                 out_channels=1,
                 dim=64,
                 num_blocks=[4, 4],
                 heads=[8, 8, 8],
                 ffn_expansion_factor=2,
                 bias=False,
                 LayerNorm_type='WithBias',
                 ):

        super(Restormer_Decoder, self).__init__()
        self.reduce_channel = nn.Conv2d(int(dim*2), int(dim), kernel_size=1, bias=bias)
        self.encoder_level2 = nn.Sequential(*[TransformerBlock(dim=dim, num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
                                            bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
        self.output = nn.Sequential(
            nn.Conv2d(int(dim), int(dim)//2, kernel_size=3,
                      stride=1, padding=1, bias=bias),
            nn.LeakyReLU(),
            nn.Conv2d(int(dim)//2, out_channels, kernel_size=3,
                      stride=1, padding=1, bias=bias),)
        self.sigmoid = nn.Sigmoid()
    def forward(self, inp_img, base_feature, detail_feature):
        out_enc_level0 = torch.cat((base_feature, detail_feature), dim=1)
        out_enc_level0 = self.reduce_channel(out_enc_level0)
        out_enc_level1 = self.encoder_level2(out_enc_level0)
        if inp_img is not None:
            out_enc_level1 = self.output(out_enc_level1) + inp_img
        else:
            out_enc_level1 = self.output(out_enc_level1)
        return self.sigmoid(out_enc_level1), out_enc_level0


def fuse_flops():
    from ptflops import get_model_complexity_info

    fusion_model = fuse_1()
    img = torch.randn(4, 1, 32, 32)
    vis_low = torch.randn(4, 1, 32, 32)
    ir_high = torch.randn(4, 1, 32, 32)
    vis_high = torch.randn(4, 1, 32, 32)
    fused_image = fusion_model(img)
    flops, params = get_model_complexity_info(fusion_model, (1, 240, 320), as_strings=True, print_per_layer_stat=True)
    print('flops: ', flops, 'params: ', params)


def debug_wavefuse():
    ir_low = torch.randn(4, 1, 32, 32)
    vis_low = torch.randn(4, 1, 32, 32)
    ir_high = torch.randn(4, 128, 32, 32)
    vis_high = torch.randn(4, 64, 32, 32)
    model = fuse_1()
    # fuse = Fusion_network()

    # en = model.encoder(ir_low)
    # f = fuse(en, en)
    # fuse_img = model.decoder_train(f)
    fuse_img = model(vis_high, vis_high, vis_high, vis_high)
    print(fuse_img.shape)

if __name__ == '__main__':
    debug_wavefuse()