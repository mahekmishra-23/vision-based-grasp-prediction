"""One small CNN and one loss function for 2D grasps."""

import torch
from torch import nn
from torch.nn import functional as F


class ConvBlock(nn.Module):
    def __init__(self, inputs, outputs):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(inputs, outputs, 3, padding=1),
            nn.BatchNorm2d(outputs), nn.ReLU(inplace=True),
            nn.Conv2d(outputs, outputs, 3, padding=1),
            nn.BatchNorm2d(outputs), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class GraspNet(nn.Module):
    """Input: RGB + mask. Output: quality, cos(2θ), sin(2θ), width maps."""

    def __init__(self):
        super().__init__()
        self.enc1 = ConvBlock(4, 16)
        self.enc2 = ConvBlock(16, 32)
        self.bottleneck = ConvBlock(32, 64)
        self.dec2 = ConvBlock(96, 32)
        self.dec1 = ConvBlock(48, 16)
        self.head = nn.Conv2d(16, 4, 1)

    def forward(self, x):
        first = self.enc1(x)
        second = self.enc2(F.max_pool2d(first, 2))
        middle = self.bottleneck(F.max_pool2d(second, 2))
        up = F.interpolate(middle, size=second.shape[-2:], mode="bilinear", align_corners=False)
        up = self.dec2(torch.cat((up, second), dim=1))
        up = F.interpolate(up, size=first.shape[-2:], mode="bilinear", align_corners=False)
        return self.head(self.dec1(torch.cat((up, first), dim=1)))


def grasp_loss(output, quality_target, regression_target, valid):
    """Train grasp location everywhere; angle and width only near labels."""
    quality = torch.sigmoid(output[:, :1])
    quality_loss = ((quality - quality_target)**2 * (1 + 8 * quality_target)).mean()

    # Encourage the highest-quality pixel to lie near an annotated grasp.
    distribution = quality_target.flatten(1)
    distribution = distribution / distribution.sum(1, keepdim=True).clamp_min(1)
    peak_loss = -(distribution * F.log_softmax(output[:, :1].flatten(1), dim=1)).sum(1).mean()

    count = valid.sum().clamp_min(1)
    angle = torch.tanh(output[:, 1:3])
    angle_loss = (((angle - regression_target[:, :2])**2) * valid).sum() / (2 * count)
    opening = torch.sigmoid(output[:, 3:4])
    width_loss = (F.smooth_l1_loss(opening, regression_target[:, 2:3], reduction="none") * valid).sum() / count
    return quality_loss + 0.1 * peak_loss + 0.3 * angle_loss + width_loss


def load_checkpoint(path):
    """The same saved weights work on a GPU or CPU laptop."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    saved = torch.load(path, map_location=device, weights_only=False)
    model = GraspNet().to(device)
    model.load_state_dict(saved["model"])
    model.eval()
    return model, int(saved["image_size"]), device
