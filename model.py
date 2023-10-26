from Net import Fusion
from Net import Parall as Pa
import torch.nn as nn
import torch
import torchvision.models as model
affine_par = True

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)
def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

# class BasicBlock(nn.Module):
#     expansion = 1
#
#     def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
#                  base_width=64, dilation=1, norm_layer=None):
#         super(BasicBlock, self).__init__()
#         if norm_layer is None:
#             norm_layer = nn.BatchNorm2d
#         if groups != 1 or base_width != 64:
#             raise ValueError('BasicBlock only supports groups=1 and base_width=64')
#         if dilation > 1:
#             raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
#         # Both self.conv1 and self.downsample layers downsample the input when stride != 1
#         self.conv1 = Pa.ModuleParallel(conv3x3(inplanes, planes, stride))
#         self.bn1 = Pa.BatchNorm2dParallel(planes,num_parallel=2)
#         self.relu = Pa.ModuleParallel(nn.ReLU(inplace=True))
#
#         self.conv2 = Pa.ModuleParallel(conv3x3(planes, planes))
#         self.bn2 = Pa.BatchNorm2dParallel(planes,num_parallel=2)
#         self.downsample = downsample
#         self.stride = stride
#
#     def forward(self, x):
#         identity = x
#
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)
#
#         out = self.conv2(out)
#         out = self.bn2(out)
#
#         if self.downsample is not None:
#             identity = self.downsample(x)
#
#
#         out = [out[l] + identity[l] for l in range(2)]
#         out = self.relu(out)
#
#         return out
#
# #
class Bottleneck(nn.Module):

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,base_width=64, dilation=1, use_shuffle=False):
        super(Bottleneck, self).__init__()

        width = int(planes * (base_width / 64.)) * groups

        self.conv1 = Pa.ModuleParallel(conv1x1(inplanes,width))
        self.bn1 = Pa.BatchNorm2dParallel(width,num_parallel=2)
        self.conv2 = Pa.ModuleParallel(conv3x3(width, width, stride, groups, dilation))
        self.bn2 = Pa.BatchNorm2dParallel(width,num_parallel=2)
        self.conv3 = Pa.ModuleParallel(conv1x1(width, planes * self.expansion))
        self.bn3 = Pa.BatchNorm2dParallel(planes * self.expansion,num_parallel=2)
        self.relu = Pa.ModuleParallel(nn.ReLU(inplace=True))
        self.downsample = downsample
        self.stride = stride
        self.use_shuffle = use_shuffle


    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)


        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)


        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = [out[l] + identity[l] for l in range(2)]
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self,groups=1, width_per_group=64):
        super(ResNet, self).__init__()

        self.inplanes = 64
        self.dilation = 1


        self.groups = groups
        self.base_width = width_per_group
        # 1
        self.conv1 = Pa.ModuleParallel(nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,bias=False))
        self.bn1   = Pa.BatchNorm2dParallel(self.inplanes,num_parallel=2)
        self.relu  = Pa.ModuleParallel(nn.ReLU(inplace=True))
        # 2
        self.maxpool = Pa.ModuleParallel(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        self.layer1 = self._make_layer(Bottleneck, 64, 3)
        # 3
        self.layer2 = self._make_layer(Bottleneck, 128,8, stride=2)
        # 4
        self.layer3 = self._make_layer(Bottleneck, 256,36, stride=2)
        # 5
        self.layer4 = self._make_layer(Bottleneck, 512,3, stride=2)

        self.ivf1 = Fusion.IVF(64)
        self.w2 = torch.nn.Parameter(torch.ones(3), requires_grad=True)
        self.ivf2 = nn.ModuleList([Fusion.IVF(256) for _ in range(3)])

        self.w3 = torch.nn.Parameter(torch.ones(8), requires_grad=True)
        self.ivf3 = nn.ModuleList([Fusion.IVF(512) for _ in range(8)])

        self.w4 = torch.nn.Parameter(torch.ones(36), requires_grad=True)
        self.ivf4 = nn.ModuleList([Fusion.IVF(1024) for _ in range(36)])

        self.w5 = torch.nn.Parameter(torch.ones(3), requires_grad=True)
        self.ivf5 = nn.ModuleList([Fusion.IVF(2048) for _ in range(3)])

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def _make_layer(self, block, planes, num_blocks, stride=1):

        downsample = None
        previous_dilation = self.dilation

        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                Pa.ModuleParallel(conv1x1(self.inplanes, planes * block.expansion, stride)),
                Pa.BatchNorm2dParallel(planes * block.expansion,num_parallel=2),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,self.base_width, previous_dilation))
        self.inplanes = planes * block.expansion
        for i in range(1, num_blocks):
            use_shuffle = False #True if i >= num_blocks - 1 else False
            layers.append(block(self.inplanes, planes))#, use_shuffle=use_shuffle))

        return nn.Sequential(*layers)

    def forward(self, x):#x[0],x[1]

        x1 = self.conv1(x)
        x1 = self.bn1(x1)
        x1 = self.relu(x1)
        Fusion1 = self.ivf1(x1[0],x1[1])

        x2 = self.maxpool(x1)
        x2 = self.layer1[0](x2)
        W2 = torch.exp(self.w2[0]) / torch.sum(torch.exp(self.w2))
        Fusion2 = W2 * self.ivf2[0](x2[0], x2[1])
        for i in range(1,len(self.layer1)):
            x2 = self.layer1[i](x2)
            W2 = torch.exp(self.w2[i]) / torch.sum(torch.exp(self.w2))
            Fusion2 += W2 * self.ivf2[i](x2[0],x2[1])

        x3 = self.layer2[0](x2)
        W3 = torch.exp(self.w3[0]) / torch.sum(torch.exp(self.w3))
        Fusion3 = W3 * self.ivf3[0](x3[0], x3[1])
        for i in range(1,len(self.layer2)):
            x3 = self.layer2[i](x3)
            W3 = torch.exp(self.w3[i]) / torch.sum(torch.exp(self.w3))
            Fusion3 += W3 * self.ivf3[i](x3[0],x3[1])

        x4 = self.layer3[0](x3)
        W4 = torch.exp(self.w4[0]) / torch.sum(torch.exp(self.w4))
        Fusion4 = W4 * self.ivf4[0](x4[0], x4[1])
        for i in range(1,len(self.layer3)):
            x4 = self.layer3[i](x4)
            W4 = torch.exp(self.w4[i]) / torch.sum(torch.exp(self.w4))
            Fusion4 += W4 * self.ivf4[i](x4[0],x4[1])


        x5 = self.layer4[0](x4)
        W5 = torch.exp(self.w5[0]) / torch.sum(torch.exp(self.w5))
        Fusion5 = W5 * self.ivf5[0](x5[0], x5[1])
        for i in range(1,len(self.layer4)):
            x5 = self.layer4[i](x5)
            W5 = torch.exp(self.w5[i]) / torch.sum(torch.exp(self.w5))
            Fusion5 += W5 * self.ivf5[i](x5[0],x5[1])

        return [Fusion1,Fusion2,Fusion3,Fusion4,Fusion5]


def expand_model_dict(model_dict, state_dict, num_parallel=2):
    model_dict_keys = model_dict.keys()
    state_dict_keys = state_dict.keys()
    for model_dict_key in model_dict_keys:
        model_dict_key_re = model_dict_key.replace('module.', '')
        if model_dict_key_re in state_dict_keys:
            model_dict[model_dict_key] = state_dict[model_dict_key_re]
        for i in range(num_parallel):
            bn = '.bn_%d' % i
            replace = True if bn in model_dict_key_re else False
            model_dict_key_re = model_dict_key_re.replace(bn, '')
            if replace and model_dict_key_re in state_dict_keys:
                model_dict[model_dict_key] = state_dict[model_dict_key_re]
    return model_dict