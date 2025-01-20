import torch
import torch.nn as nn
import torch.nn.functional as F
from tools.evaluator import Evaluator



class Fusionloss(nn.Module):
    def __init__(self):
        super(Fusionloss, self).__init__()
        self.sobelconv=Sobelxy()

    def forward(self,vis_rgb,image_ir,generate_img):
        image_lab = self.rgbs_trans(vis_rgb)
        image_l=image_lab[:,:1,:,:]
        x_in_max=torch.max(image_l,image_ir)
        loss_in=F.l1_loss(x_in_max,generate_img)
        y_grad=self.sobelconv(image_l)
        ir_grad=self.sobelconv(image_ir)
        generate_img_grad=self.sobelconv(generate_img)
        x_grad_joint=torch.max(y_grad,ir_grad)
        loss_grad=F.l1_loss(x_grad_joint,generate_img_grad)
        loss_total=loss_in+10*loss_grad
        return loss_total,loss_in,loss_grad

    def rgbs_trans(self, images):
        return torch.stack([self.rgb2yuv(img) for img in images])

    def rgb2lab(self, image):
        # Convert RGB to XYZ
        def rgb_to_xyz(rgb):
            # Linearize sRGB values
            mask = rgb > 0.04045
            rgb = torch.where(mask, ((rgb + 0.055) / 1.055) ** 2.4, rgb / 12.92)

            # RGB to XYZ matrix (standard D65 illuminant)
            rgb_to_xyz_matrix = torch.tensor([
                [0.4124564, 0.3575761, 0.1804375],
                [0.2126729, 0.7151522, 0.0721750],
                [0.0193339, 0.1191920, 0.9503041],
            ], dtype=torch.float32).to(image.device)

            # Reshape and transform the RGB image
            rgb_permuted = rgb.permute(1, 2, 0)  # Change to HWC
            xyz_image = torch.tensordot(rgb_permuted, rgb_to_xyz_matrix, dims=([2], [1]))  # Transform
            return xyz_image.permute(2, 0, 1)  # Back to CHW

        # Convert XYZ to Lab
        def xyz_to_lab(xyz):
            # Normalize for D65 white point
            xyz_ref_white = torch.tensor([0.95047, 1.0, 1.08883], dtype=torch.float32).view(3, 1, 1).to(xyz.device)
            xyz = xyz / xyz_ref_white

            # f(t) function for Lab conversion
            epsilon = 0.008856
            kappa = 903.3
            mask = xyz > epsilon
            xyz = torch.where(mask, xyz ** (1 / 3), (kappa * xyz + 16) / 116)

            # Convert to Lab
            L = 116 * xyz[1] - 16
            a = 500 * (xyz[0] - xyz[1])
            b = 200 * (xyz[1] - xyz[2])

            return torch.stack([L, a, b])

        # Perform RGB to XYZ, then XYZ to Lab
        xyz = rgb_to_xyz(image)
        lab = xyz_to_lab(xyz)

        return lab

    def rgb2yuv(self, image):
        rgb_to_yuv_matrix = torch.tensor([
            [0.299, 0.587, 0.114],
            [-0.14713, -0.28886, 0.436],
            [0.615, -0.51499, -0.10001]
        ], dtype=torch.float32).to(image.device)
        image_permuted = image.permute(1, 2, 0)
        yuv_image = torch.tensordot(image_permuted, rgb_to_yuv_matrix, dims=([2], [1]))
        return yuv_image.permute(2, 0, 1)


class AdaptiveLossWeighting(nn.Module):
    '''
    loss_count: num of loss part
    losses: [loss1, loss2, ...]
    '''
    def __init__(self, loss_count):
        super(AdaptiveLossWeighting, self).__init__()
        self.loss_count = loss_count
        self.logits = nn.Parameter(torch.ones(self.loss_count))

    def forward(self, losses):
        assert len(losses) == self.loss_count
        loss_weights = F.softmax(self.logits, dim=0)
        weighted_loss = torch.sum(loss_weights * losses)
        return weighted_loss


class Sobelxy(nn.Module):
    def __init__(self):
        super(Sobelxy, self).__init__()
        kernelx = [[-1, 0, 1],
                  [-2,0 , 2],
                  [-1, 0, 1]]
        kernely = [[1, 2, 1],
                  [0,0 , 0],
                  [-1, -2, -1]]
        kernelx = torch.FloatTensor(kernelx).unsqueeze(0).unsqueeze(0)
        kernely = torch.FloatTensor(kernely).unsqueeze(0).unsqueeze(0)
        self.weightx = nn.Parameter(data=kernelx, requires_grad=False).cuda()
        self.weighty = nn.Parameter(data=kernely, requires_grad=False).cuda()
    def forward(self,x):
        sobelx=F.conv2d(x, self.weightx, padding=1)
        sobely=F.conv2d(x, self.weighty, padding=1)
        return torch.abs(sobelx)+torch.abs(sobely)


class AFDLoss(nn.Module):
    def __init__(self):
        super(AFDLoss, self).__init__()

    def forward(self, ir_frequency, vis_frequency):

        ir_frequency = [ir.to('cpu').numpy() for ir in ir_frequency]
        vis_frequency = [vis.to('cpu').numpy() for vis in vis_frequency]
        ir_low, ir_high = ir_frequency
        vis_low, vis_high = vis_frequency
        low_loss_ir = low_loss_vis = high_loss_ir = high_loss_vis = 0
        for i in range(len(ir_low)):
            # low
            low_loss_ir += 0.5 * (1 / (Evaluator.EN(ir_low[i][0])+1) + 1 / (Evaluator.SD(ir_low[i][0])+1))
            low_loss_vis += 0.5 * (1 / (Evaluator.EN(vis_low[i][0])+1) + 1 / (Evaluator.SD(vis_low[i][0])+1))
            # high
            high_loss_ir += 0.5 * (1 / (Evaluator.AG(ir_high[i][0])+1) + 1 / (Evaluator.SF(ir_high[i][0])+1))
            high_loss_vis += 0.5 * (1 / (Evaluator.AG(vis_high[i][0])+1) + 1 / (Evaluator.SF(vis_high[i][0])+1))

        return 0.25 * (low_loss_ir + low_loss_vis + high_loss_ir + high_loss_vis) / len(ir_low)

def cc(img1, img2):
    eps = torch.finfo(torch.float32).eps
    """Correlation coefficient for (N, C, H, W) image; torch.float32 [0.,1.]."""
    N, C, _, _ = img1.shape
    img1 = img1.reshape(N, C, -1)
    img2 = img2.reshape(N, C, -1)
    img1 = img1 - img1.mean(dim=-1, keepdim=True)
    img2 = img2 - img2.mean(dim=-1, keepdim=True)
    cc = torch.sum(img1 * img2, dim=-1) / (eps + torch.sqrt(torch.sum(img1 **
                                                                      2, dim=-1)) * torch.sqrt(torch.sum(img2**2, dim=-1)))
    cc = torch.clamp(cc, -1., 1.)
    return cc.mean()