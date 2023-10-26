import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class FocalLoss(nn.Module):
    def __init__(self, class_num, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()

        self.gamma = gamma  # 指数
        self.class_num = class_num  # 类别数目
        self.size_average = size_average  # 返回的loss是否需要mean一下

    def compute_class_weights(self, histogram):

        classWeights = np.ones(self.class_num, dtype=np.float32)
        normHist = histogram / np.sum(histogram)

        for i in range(self.class_num):
            classWeights[i] = 1 / (np.log(1.10 + normHist[i]))

        # classWeights[0] = 0.0

        return classWeights

    def forward(self, inputs, targets):

        input = inputs[0,:,:,:]
        target = targets.contiguous().view(-1)

        # 统计各类别像素点个数
        number_0 = torch.sum(target == 0).item()
        number_1 = torch.sum(target == 1).item()
        number_2 = torch.sum(target == 2).item()
        number_3 = torch.sum(target == 3).item()
        number_4 = torch.sum(target == 4).item()
        number_5 = torch.sum(target == 5).item()
        number_6 = torch.sum(target == 6).item()
        number_7 = torch.sum(target == 7).item()
        number_8 = torch.sum(target == 8).item()

        # print(number_0, number_1, number_2, number_3, number_4, number_5, number_6, number_7, number_8)
        frequency = torch.tensor((number_0, number_1, number_2, number_3, number_4, number_5, number_6, number_7, number_8),dtype=torch.float32)
        frequency = frequency.numpy()
        classWeights = self.compute_class_weights(frequency)
        alpha = torch.from_numpy(classWeights).float().cuda()
        alpha = torch.unsqueeze(alpha,dim=-1)

        # target : N, 1, H, W
        inputs = inputs.permute(0, 2, 3, 1)
        targets = torch.unsqueeze(targets,dim=1).permute(0, 2, 3, 1)

        num, h, w, C = inputs.size()
        N = num * h * w
        inputs = inputs.reshape(N, -1)   # N, C
        targets = targets.reshape(N, -1)  # 待转换为one hot label

        P = F.softmax(inputs, dim=1)  # 先求p_t
        class_mask = inputs.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)  # 得到label的one_hot编码

        if inputs.is_cuda and not alpha.is_cuda:
            alpha = alpha.cuda()  # 如果是多GPU训练 这里的cuda要指定搬运到指定GPU上 分布式多进程训练除外
        alpha = alpha[ids.data.view(-1)]

        probs = (P * class_mask).sum(1).view(-1, 1)
        log_p = probs.log()
        G = (torch.pow((1 - probs), self.gamma))

        # a = alpha[alpha != 0.0]
        # g = G    [alpha != 0.0]
        # l = log_p[alpha != 0.0]

        batch_loss = - alpha * G * log_p

        if self.size_average: loss = batch_loss.mean()
        else:  loss = batch_loss.sum()

        return loss

class Edge_FocalLoss(nn.Module):
    def __init__(self, class_num, gamma=2, size_average=True):
        super(Edge_FocalLoss, self).__init__()

        self.gamma = gamma  # 指数
        self.class_num = class_num  # 类别数目
        self.size_average = size_average  # 返回的loss是否需要mean一下

    def compute_class_weights(self, histogram):

        classWeights = np.ones(self.class_num, dtype=np.float32)
        normHist = histogram / np.sum(histogram)

        for i in range(self.class_num):
            classWeights[i] = 1 / (np.log(1.10 + normHist[i]))

        # classWeights[0] = 0.0

        return classWeights

    def forward(self, inputs, targets):

        input = inputs[0,:,:,:]
        target = targets.contiguous().view(-1)

        # 统计各类别像素点个数
        number_0 = torch.sum(target == 0).item()
        number_1 = torch.sum(target == 1).item()

        # print(number_0, number_1, number_2, number_3, number_4, number_5, number_6, number_7, number_8)
        frequency = torch.tensor((number_0, number_1),dtype=torch.float32)
        frequency = frequency.numpy()
        classWeights = self.compute_class_weights(frequency)
        alpha = torch.from_numpy(classWeights).float().cuda()
        alpha = torch.unsqueeze(alpha,dim=-1)

        # target : N, 1, H, W
        inputs = inputs.permute(0, 2, 3, 1)
        targets = torch.unsqueeze(targets,dim=1).permute(0, 2, 3, 1)

        num, h, w, C = inputs.size()
        N = num * h * w
        inputs = inputs.reshape(N, -1)   # N, C
        targets = targets.reshape(N, -1)  # 待转换为one hot label

        P = F.softmax(inputs, dim=1)  # 先求p_t
        class_mask = inputs.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)  # 得到label的one_hot编码

        if inputs.is_cuda and not alpha.is_cuda:
            alpha = alpha.cuda()  # 如果是多GPU训练 这里的cuda要指定搬运到指定GPU上 分布式多进程训练除外
        alpha = alpha[ids.data.view(-1)]

        probs = (P * class_mask).sum(1).view(-1, 1)
        log_p = probs.log()
        G = (torch.pow((1 - probs), self.gamma))


        batch_loss = - alpha * G * log_p

        if self.size_average: loss = batch_loss.mean()
        else:  loss = batch_loss.sum()

        return loss

# class ImageBasedCrossEntropyLoss2d(nn.Module):
#
#     def __init__(self, classes, weight=None, size_average=True, ignore_index=255,
#                  norm=False, upper_bound=1.0):
#         super(ImageBasedCrossEntropyLoss2d, self).__init__()
#
#         self.num_classes = classes
#         self.nll_loss = nn.NLLLoss2d(weight, size_average, ignore_index)
#         self.norm = norm
#         self.upper_bound = upper_bound
#
#     def calculateWeights(self, target):
#
#         hist = np.histogram(target.flatten(), range(self.num_classes+1), normed=True)[0]
#         hist = ((hist != 0) * self.upper_bound * (1 - hist)) + 1
#         return hist
#
#     def forward(self, inputs, targets):
#
#         target_cpu = targets.data.cpu().numpy()
#
#         weights = self.calculateWeights(target_cpu)
#         self.nll_loss.weight = torch.Tensor(weights).cuda()
#
#         print(self.nll_loss.weight.shape)
#         loss = 0.0
#
#         for i in range(0, inputs.shape[0]):
#             loss += self.nll_loss(F.log_softmax(inputs[i].unsqueeze(0)),targets[i].unsqueeze(0))
#
#         return loss.mean()
#

# # #
# class BCEFocalLoss(torch.nn.Module):
#     def __init__(self, alpha=1, gamma=2, logits=False, reduce=True):
#         super(BCEFocalLoss, self).__init__()
#         self.alpha = alpha
#         self.gamma = gamma
#         self.logits = logits
#         self.reduce = reduce
#
#     def forward(self, inputs, targets):
#
#         BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduce=False)
#         pt = torch.exp(-BCE_loss)
#         print(pt.shape)
#         F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
#         return torch.mean(F_loss)


class BCEFocalLoss(nn.Module):
    def __init__(self, class_num, gamma=2, size_average=True):
        super(BCEFocalLoss, self).__init__()

        self.gamma = gamma  # 指数
        self.class_num = class_num  # 类别数目
        self.size_average = size_average  # 返回的loss是否需要mean一下

    def compute_class_weights(self, histogram):
        classWeights = np.ones(self.class_num, dtype=np.float32)
        normHist = histogram / np.sum(histogram)
        for i in range(self.class_num):
            classWeights[i] = 1 / (np.log(1.10 + normHist[i]))
        return classWeights

    def forward(self, inputs, targets):


        input = inputs[0,:,:,:]
        target = targets.contiguous().view(-1)

        # 统计各类别像素点个数
        number_0 = torch.sum(target == 0).item()
        number_1 = torch.sum(target == 1).item()

        # print(number_0, number_1, number_2, number_3, number_4, number_5, number_6, number_7, number_8)
        frequency = torch.tensor((number_0, number_1),dtype=torch.float32)
        frequency = frequency.numpy()
        classWeights = self.compute_class_weights(frequency)
        alpha = torch.from_numpy(classWeights).float().cuda()
        alpha = torch.unsqueeze(alpha,dim=-1)

        # target : N, 1, H, W
        inputs = inputs.permute(0, 2, 3, 1)
        targets = torch.unsqueeze(targets,dim=1).permute(0, 2, 3, 1)

        num, h, w, C = inputs.size()
        N = num * h * w
        inputs = inputs.reshape(N, -1)   # N, C
        targets = targets.reshape(N, -1)  # 待转换为one hot label
        P = F.softmax(inputs, dim=1)  # 先求p_t
        class_mask = inputs.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)  # 得到label的one_hot编码

        if inputs.is_cuda and not alpha.is_cuda:
            alpha = alpha.cuda()  # 如果是多GPU训练 这里的cuda要指定搬运到指定GPU上 分布式多进程训练除外
        alpha = alpha[ids.data.view(-1)]

        probs = (P * class_mask).sum(1).view(-1, 1)
        log_p = probs.log()
        G = (torch.pow((1 - probs), self.gamma))

        batch_loss = - alpha * G * log_p

        if self.size_average: loss = batch_loss.mean()
        else:  loss = batch_loss.sum()

        return loss

#
#
def bce2d(input, target):
    n, c, h, w = input.size()

    log_p = input.transpose(1, 2).transpose(2, 3).contiguous().view(1, -1)
    target_t = target.transpose(1, 2).transpose(2, 3).contiguous().view(1, -1)
    target_trans = target_t.clone()

    pos_index = (target_t == 1)
    neg_index = (target_t == 0)
    ignore_index = (target_t > 1)

    target_trans[pos_index] = 1
    target_trans[neg_index] = 0

    pos_index = pos_index.data.cpu().numpy().astype(bool)
    neg_index = neg_index.data.cpu().numpy().astype(bool)
    ignore_index = ignore_index.data.cpu().numpy().astype(bool)

    weight = torch.Tensor(log_p.size()).fill_(0)
    weight = weight.numpy()
    pos_num = pos_index.sum()
    neg_num = neg_index.sum()
    sum_num = pos_num + neg_num

    weight[pos_index] = (neg_num * 1.0 / sum_num)*((1 - torch.sigmoid(log_p).cpu().detach().numpy()) ** 2)
    weight[neg_index] = (pos_num * 1.0 / sum_num)*(torch.sigmoid(log_p).cpu().detach().numpy() ** 2)

    weight[ignore_index] = 0
    weight = torch.from_numpy(weight)
    weight = weight.cuda()

    loss = F.binary_cross_entropy_with_logits(log_p, target_t, weight, size_average=True)

    return loss