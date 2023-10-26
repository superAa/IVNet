
import torch
import torch.nn.functional as F
from torch import nn
import torchvision

from Net import Fusion
from Net import model
k = 64

class JLModule(nn.Module):
    def __init__(self):
        super(JLModule, self).__init__()

        self.relu = nn.ReLU(inplace=True)
        # 用Vgg第一层的卷积来填充
        self.vgg_rgb_conv1 = nn.Sequential(nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1), nn.ReLU(inplace=True),
                                           nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),nn.ReLU(inplace=True))
        self.vgg_rgb_conv1.load_state_dict(torch.load('/vgg_conv1.pth'), strict=True)
        self.resnet_raw_model = model.ResNet()

        cached_file = '/resnet50-19c8e357.pth'
        state_dict = torch.load(cached_file, map_location=None)
        model_dict = model.expand_model_dict(self.resnet_raw_model.state_dict(), state_dict, num_parallel=2)
        self.resnet_raw_model.load_state_dict(model_dict, strict=True)

        self.ivf0 = Fusion.IVF(64)

    def forward(self, rgb,thermal):

        # 存储特征，有
        x = [rgb,thermal]

        rt_feature = []
        r_feature = self.vgg_rgb_conv1(rgb)
        t_feature = self.vgg_rgb_conv1(thermal)
        Fusion0 = self.ivf0(r_feature, t_feature)

        rt_feature.append(Fusion0)

        RT_feature = self.resnet_raw_model(x)

        out = rt_feature + RT_feature

        return out



class FAModule(nn.Module):
    def __init__(self):
        super(FAModule, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        #------------------------
        self.conv_branch1 = nn.Sequential(nn.Conv2d(k, int(k / 4), 1))
        self.conv_branch11  = nn.Sequential(nn.BatchNorm2d(int(k / 4)), self.relu)
        # -----------------------
        self.conv_branch2 = nn.Sequential(nn.Conv2d(k, int(k / 2), 1), self.relu,
                                          nn.Conv2d(int(k / 2), int(k / 4), 3, 1, 1))
        self.conv_branch21  = nn.Sequential(nn.BatchNorm2d(int(k / 4)), self.relu)
        # -----------------------
        self.conv_branch3 = nn.Sequential(nn.Conv2d(k, int(k / 4), 1), self.relu,
                                          nn.Conv2d(int(k / 4), int(k / 4), 5, 1, 2))
        self.conv_branch31  = nn.Sequential(nn.BatchNorm2d(int(k / 4)), self.relu)
        # -----------------------
        self.conv_branch4 = nn.Sequential(nn.MaxPool2d(3, 1, 1), nn.Conv2d(k, int(k / 4), 1))
        self.conv_branch41  = nn.Sequential(nn.BatchNorm2d(int(k / 4)), self.relu)


    def forward(self, x,Flag=0):

        # aggregation
        x_branch1 = self.conv_branch1(x)
        x_branch1 = self.conv_branch11(x_branch1)

        x_branch2 = self.conv_branch2(x)
        x_branch2 = self.conv_branch21(x_branch2)

        x_branch3 = self.conv_branch3(x)
        x_branch3 = self.conv_branch31(x_branch3)

        x_branch4 = self.conv_branch4(x)
        x_branch4 = self.conv_branch41(x_branch4)

        x = torch.cat((x_branch1, x_branch2, x_branch3, x_branch4), dim=1)

        return x

class FANet(nn.Module):

    def __init__(self):
        super(FANet, self).__init__()
        self.FA1 = FAModule()
        self.FA2 = FAModule()
        self.FA3 = FAModule()
        self.FA4 = FAModule()
        self.FA5 = FAModule()

    def forward(self,x):

        # 1
        x[5] = F.interpolate(x[5], scale_factor=2, mode='bilinear', align_corners=True)
        xa5 = self.FA5(x[4] + x[5])
        # 2
        xa5_4 = F.interpolate(xa5, scale_factor=2, mode='bilinear', align_corners=True)
        x[5]  = F.interpolate(x[5], scale_factor=2, mode='bilinear', align_corners=True)
        xa4  = self.FA4(xa5_4 + x[3] + x[5])
        # 3
        xa4_3 = F.interpolate(xa4,   scale_factor=2,  mode='bilinear', align_corners=True)
        xa5_3 = F.interpolate(xa5_4, scale_factor=2,  mode='bilinear', align_corners=True)
        x[5]  = F.interpolate(x[5],  scale_factor=2, mode='bilinear', align_corners=True)
        xa3   = self.FA3(xa4_3 + x[2] + xa5_3 + x[5])
        # 4
        xa3_2 = F.interpolate(xa3,    scale_factor=2,  mode='bilinear', align_corners=True)
        xa4_2 = F.interpolate(xa4_3,  scale_factor=2,  mode='bilinear', align_corners=True)
        xa5_2 = F.interpolate(xa5_3,  scale_factor=2,  mode='bilinear', align_corners=True)
        x[5]  = F.interpolate(x[5],   scale_factor=2,  mode='bilinear', align_corners=True)
        xa2  = self.FA2(xa3_2 + x[1] + xa4_2 + xa5_2 + x[5])
        # 5
        xa2_1 = F.interpolate(xa2,  scale_factor=2,   mode='bilinear', align_corners=True)
        xa3_1 = F.interpolate(xa3_2, scale_factor=2,  mode='bilinear', align_corners=True)
        xa4_1 = F.interpolate(xa4_2, scale_factor=2,  mode='bilinear', align_corners=True)
        xa5_1 = F.interpolate(xa5_2, scale_factor=2,  mode='bilinear', align_corners=True)
        x[5]  = F.interpolate(x[5],  scale_factor=2,  mode='bilinear', align_corners=True)
        xa1 = self.FA1(xa2_1 + x[0] + xa3_1 + xa4_1 + xa5_1 + x[5])

        return xa1


class GSCNN(nn.Module):

    def __init__(self, num_classes):

        super(GSCNN, self).__init__()
        # 输出是长度为6的列表: 640,320,160,80,40,80
        self.JL = JLModule()
        # 对应整合输出64的通道
        channel = [64,64,256,512,1024,2048]
        self.cp = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(channel[i],64,1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True)
                ) for i in range(6)
            ]
        )

        # 分割聚合
        self.FA1 = FANet()

        # 分割
        self.seg_out = nn.Sequential( nn.Conv2d(64,num_classes,1),nn.BatchNorm2d(num_classes) )
        self.edge_conv = nn.Sequential( nn.Conv2d(num_classes,1,1) )

    def forward(self, rgb,thermal):

        seg,useg = [],[]

        # r是长度6的RGB特征列表，t是长度6的热红外特征列表, 0-5

        RT = self.JL(rgb, thermal)#2
        for i in range(6):
            RT[i] = self.cp[i](RT[i])

        # 从深到浅,密集整合。

        seg_out = self.FA1(RT)
        seg_out = self.seg_out(seg_out)
        edge_out = self.edge_conv(seg_out)

        return seg_out,edge_out