import torch
import torch.nn as nn
import torchvision.models as models

class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        #加载VGG19
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features

        #取出前35层
        self.vgg_layers = nn.Sequential(*list(vgg.children())[:35])

        for param in self.vgg_layers.parameters():
            param.requires_grad = False

        self.criterion = nn.L1Loss()

    def forward(self, generated, ground_truth):
        mean = torch.tensor([0.485, 0.456, 0.406], device=generated.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=generated.device).view(1, 3, 1, 1)

        generated = (generated - mean) / std
        ground_truth = (ground_truth - mean) / std

        gen_features = self.vgg_layers(generated)
        gt_features  = self.vgg_layers(ground_truth)

        loss = self.criterion(gen_features, gt_features)
        return loss
