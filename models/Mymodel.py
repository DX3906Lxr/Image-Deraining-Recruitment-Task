import torch
import torch.nn as nn
import torch.nn.functional as F
"""
参考了aster学长给的前两篇论文，分别是PRN和SpaNet
我这模型挺简单，SpaNet把注意力图加在多残差块堆叠的深层网络，而我将注意力图加在PRN渐进式网络中
或者可以说就是“精简”SpaNet，然后多阶段学习
于是在每个stage中先用一个卷积来拓展通道，学习局部特征，ReLU激活
进入一个残差块来学习特征，生成“特征图”，然后进入
和SpaNet中一模一样的Spatial Attentive Residual Block(SARB)
在这里有两个分支，一个分支中特征图进入一个ResidualBlock残差块进一步提取特征
而另一个注意力分支就负责捕捉空间上下文信息，对四个方向生成相应的权重（方向权重），从而选择性地突出投影的降雨特征
然后融合，再来一个一个残差块重建干净背景
然后循环，进入下一阶段
"""

def conv1x1(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, 1, stride, padding=0, bias=False)

def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, 3, stride, padding=1, bias=False)

#就是一个普通的残差块
class Residualblock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Residualblock, self).__init__()
        self.group = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=2, bias=False, dilation=2),
            nn.ReLU(inplace=False),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=2, bias=False, dilation=2),
            nn.ReLU(inplace=False),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=2, bias=False, dilation=2)
        )

    def forward(self, x):
        return self.group(x) + x

"""
用和第二篇论文中一样的两轮提取到全局的方向信息
IRNN 严格对应论文公式：
h_{i,j} ← max( α_dir * h_{i,j-1} + h_{i,j}, 0 )
沿四个方向（上下左右）逐像素递推
"""
class Spacial_IRNN(nn.Module):

    def __init__(self, in_channels, alpha=0.2):
        super(Spacial_IRNN, self).__init__()
        #在每个通道上定义一个可学习的标量参数 α，初始值由为alpha
        self.alpha = nn.Parameter(torch.ones(in_channels) * alpha, requires_grad=True)

    def forward(self, x):#x: [B, C, H, W]
        B, C, H, W = x.shape

        # 通过 .view() 显式增加维度
        # self.alpha.view(1, C, 1) 会把原来 [C] 的张量变成 三维张量 [1, C, 1]
        alpha = self.alpha.view(1, C, 1)

        # h_dt = torch.zeros_like(x)
        # for i in reversed(range(H)):
        #     if i == H - 1:
        #         h_dt[:, :, i, :] = F.relu(x[:, :, i, :])
        #     else:
        #         h_dt[:, :, i, :] = F.relu(self.alpha.view(1, C, 1, 1) * h_dt[:, :, i + 1, :] + x[:, :, i, :])

        # 上到下投影
        """
        我开始的时候是用的上面注释掉的代码
        forward 过程中h_td被原地修改，这会使梯度在 backward 时丢失
        """
        #创建一个空列表，用于按行收集每个位置的输出结果
        h_td = []
        #初始化“上一行的隐藏状态”，取一个形状与输入的单行相同的零张量
        prev = torch.zeros_like(x[:, :, 0, :])
        for i in range(H):
            # 第一个像素没有前一项，潜变量单独处理
            cur = F.relu(x[:, :, i, :] + alpha * prev if i > 0 else x[:, :, i, :])
            h_td.append(cur)
            #更新“上一行”结果。下一个循环时，这一行就变成下一行的“上方特征
            prev = cur
        h_td = torch.stack(h_td, dim=2)  # [B, C, H, W]

        #上到下
        h_dt = []
        #取出最后一行
        prev = torch.zeros_like(x[:, :, -1, :])
        for i in reversed(range(H)):
            cur = F.relu(x[:, :, i, :] + alpha * prev if i < H - 1 else x[:, :, i, :])
            h_dt.append(cur)
            prev = cur
        h_dt.reverse()
        h_dt = torch.stack(h_dt, dim=2)

        # 、左到右
        h_lr = []
        prev = torch.zeros_like(x[:, :, :, 0])
        for j in range(W):
            cur = F.relu(x[:, :, :, j] + alpha * prev if j > 0 else x[:, :, :, j])
            h_lr.append(cur)
            prev = cur
        h_lr = torch.stack(h_lr, dim=3)

        # 右到左
        h_rl = []
        prev = torch.zeros_like(x[:, :, :, -1])
        for j in reversed(range(W)):
            cur = F.relu(x[:, :, :, j] + alpha * prev if j < W - 1 else x[:, :, :, j])
            h_rl.append(cur)
            prev = cur
        h_rl.reverse()
        h_rl = torch.stack(h_rl, dim=3)

        return h_td, h_lr, h_dt, h_rl

#注意力分支 也就是方向的权重生成，从而选择性地突出投影的降雨特征
class Attention(nn.Module):
    def __init__(self, in_channels):
        super(Attention, self).__init__()
        #特征压缩，通道减半
        mid = in_channels // 2
        self.conv1 = conv3x3(in_channels, mid)
        self.relu1 = nn.ReLU(inplace=False)
        self.conv2 = conv3x3(mid, mid)
        self.relu2 = nn.ReLU(inplace=False)
        self.conv3 = nn.Conv2d(mid, 4, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.relu1(self.conv1(x))
        out = self.relu2(self.conv2(out))
        out = self.conv3(out)
        return self.sigmoid(out)  # [B,4,H,W]


#SAM生成方向注意力图 (两轮IRNN + 方向权重)
class SAM(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SAM, self).__init__()
        self.irnn1 = Spacial_IRNN(out_channels)
        self.irnn2 = Spacial_IRNN(out_channels)
        self.conv_in = conv3x3(in_channels, in_channels)
        self.conv2 = conv3x3(in_channels * 4, in_channels)
        self.conv3 = conv3x3(in_channels * 4, in_channels)
        self.relu = nn.ReLU(inplace=False)
        self.att = Attention(in_channels)
        self.conv_out = conv1x1(out_channels, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        weight = self.att(x)  # [B,4,H,W]
        out = self.conv_in(x)
        up, right, down, left = self.irnn1(out)

        # 第一次方向加权，weight保留原维度
        up    = up    * weight[:,0:1,:,:]
        right = right * weight[:,1:2,:,:]
        down  = down  * weight[:,2:3,:,:]
        left  = left  * weight[:,3:4,:,:]

        out = torch.cat([up, right, down, left], dim=1)
        out = self.conv2(out)
        up, right, down, left = self.irnn2(out)

        # 第二次方向加权
        up    = up    * weight[:,0:1,:,:]
        right = right * weight[:,1:2,:,:]
        down  = down  * weight[:,2:3,:,:]
        left  = left  * weight[:,3:4,:,:]

        out = torch.cat([up, right, down, left], dim=1)
        out = self.relu(self.conv3(out))
        mask = self.sigmoid(self.conv_out(out))  # [B,1,H,W]
        return mask


#阶段函数，每个重复的渐进式的网络都是层数精简版的SPANet
class SPANetStage(nn.Module):
    def __init__(self, ch=32):
        super(SPANetStage, self).__init__()
        self.conv_in = nn.Sequential(conv3x3(6, ch), nn.ReLU(inplace=False
                                                             ))  # rainy+prev
        self.sam = SAM(ch, ch)
        self.res = Residualblock(ch, ch)
        self.conv = nn.Sequential(
            nn.Conv2d(ch, ch, kernel_size=1, bias=False),
            nn.ReLU(inplace=False),
            nn.Conv2d(ch, ch, kernel_size=3, stride=1, padding=2, bias=False, dilation=2),
            nn.ReLU(inplace=False),
            nn.Conv2d(ch, ch, kernel_size=1, bias=False)
        )
        self.conv_out = conv3x3(ch, 3)

    """
    原论文中的SpaNet是端到端直接生成的去雨图，这里采用的是残差的思想，用SpaNet生成雨层
    以及保持PRN中每个阶段的输入等于上个阶段的输入再与原图拼接的思路
    """
    def forward(self, rainy, clean_prev):
        x = torch.cat([rainy, clean_prev], dim=1)
        #得到特征图，然后就进入“SAB”
        feat = self.res(self.conv_in(x))
        #进入SAB
        attn = self.sam(feat)
        #两者相加，这里也是残差的思想，这就是Spatial Attentive Residual Block (SARB)
        out = F.relu(self.conv(feat) * attn + feat)
        delta = self.conv_out(out)
        clean = torch.clamp(rainy + delta, 0.0, 1.0)
        return clean, attn


#渐进式多阶段SPANet
class Mymodel(nn.Module):
    def __init__(self, stages=3, ch=32):
        super(Mymodel, self).__init__()
        self.stages = stages
        self.stage = SPANetStage(ch)  # 参数共享

    def forward(self, rainy):
        clean = rainy.clone()
        for _ in range(self.stages):
            clean, attn = self.stage(rainy, clean)
        return clean


