# losses/easy_contrastive_loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class EasyContrastiveLoss(nn.Module):
    """
    使用 VGG19(前35层) 的若干中间层作为特征，
    自适应平均池化到 (N，C,1,1)
    再拼成一条特征向量
    余弦相似度部分：
      - 正样本：pred vs gt => 最大化相似度 -> loss_pos = 1 - cos
      - 负样本：pred vs inp => 最小化相似度 -> loss_neg = ReLU(cos) 让其尽量 ≤ 0
    总损失：loss = loss_pos + loss_neg
    """
    def __init__(self, selected_ids=(3, 8, 17, 26, 35)):
        super().__init__()
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features
        self.vgg_layers = nn.Sequential(*list(vgg.children())[:35])

        # 挑选的中间层，创立一个集合（与常见 perceptual 的多层取法一致：conv1_2, conv2_2, conv3_4, conv4_4, conv5_4）
        self.selected_ids = set(selected_ids)

        # 冻结 VGG 参数
        for p in self.vgg_layers.parameters():
            p.requires_grad = False
        #AdaptiveAvgPool2d可以直接指定输出的尺寸大小
        #我们自适应池化到 1x1，把任意 HxW 的特征图压成向量
        #从N，C，H，W压缩到N x C x 1 x 1，其实就是对每个通道特征图求总的特征值
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # VGG 标准化的 mean/std（ImageNet 统计）
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.register_buffer("std",  torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1))

    @torch.no_grad()
    def _normalize_for_vgg(self, x: torch.Tensor) -> torch.Tensor:
        """
        按照ImageNet 统计的 mean/std 标准化
        """
        # 如果输入是 0~255（uint8 或 float）
        if x.dtype != torch.float32 and x.dtype != torch.float64:
            x = x.float()
        if torch.max(x) > 1.0:
            x = x / 255.0

        # ImageNet 均值与方差
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)

        x = (x - mean) / std
        return x

    def _extract_feature_vector(self, x: torch.Tensor) -> torch.Tensor:
        """
        输入: N x C x H x W
        输出: N x D  (将若干层的池化向量拼接)
        """
        feats = []
        h = x
        #相当于用这个for循环手动forward预训练模型，来提取特征图
        for i, layer in enumerate(self.vgg_layers):
            h = layer(h)
            #如果是我们挑选的层
            if (i + 1) in self.selected_ids:
                # 先用池化到N x C x 1 x 1 再展平成 N x C（从第一维展开）
                pooled = self.pool(h).flatten(1)
                feats.append(pooled)
        # 沿通道维拼接成一条向量
        return torch.cat(feats, dim=1)

    def forward(self, pred: torch.Tensor, gt: torch.Tensor, inp: torch.Tensor) -> torch.Tensor:
        # 按 VGG 预训练规则做标准化（不会回传梯度）
        pred_n = self._normalize_for_vgg(pred)
        gt_n   = self._normalize_for_vgg(gt)
        inp_n  = self._normalize_for_vgg(inp)

        # 抽取并压缩为特征向量
        f_pred = self._extract_feature_vector(pred_n)  # N x D
        f_gt   = self._extract_feature_vector(gt_n)    # N x D
        f_inp  = self._extract_feature_vector(inp_n)   # N x D

        # 计算余弦相似度（逐样本）
        cos_pos = F.cosine_similarity(f_pred, f_gt, dim=1)   # 越大越好
        cos_neg = F.cosine_similarity(f_pred, f_inp, dim=1)  # 越小越好（最好 ≤ 0）

        # 损失设计：
        #   正样本：1 - cos_pos -> 让pred与gt相近
        #   负样本：ReLU(cos_neg) -> 惩罚相似为正的情况，鼓励 <= 0
        loss_pos = 1.0 - cos_pos
        loss_neg = F.relu(cos_neg)

        loss = (loss_pos + loss_neg).mean()
        return loss
