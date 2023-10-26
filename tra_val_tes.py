
import torch
import torch.nn.functional as F
import numpy as np
from Loss.DualTaskLoss import DualTaskLoss
from Loss.Lovasz_Loss import lovasz_softmax,lovasz_hinge
from Loss.Focal_loss import FocalLoss,Edge_FocalLoss
from sklearn.metrics import confusion_matrix
from Dataset.util import compute_results,visualize
from Net import net
from torch.autograd import Variable
from torch.utils.data import DataLoader
from Dataset.MyDataset import MF_dataset0
import cv2

def Edge_attention( input2, target2, edge2):

    filler = torch.ones_like(target2) * 255
    return F.cross_entropy(input2, torch.where(edge2.max(1)[0] == 1.0, target2, filler), ignore_index=255)

from torch.nn.utils import clip_grad_norm_
def train(
        train_dataset,
        train_loader, # 训练数据载入
        net, # 网络
        viz,
        optimizer, # 优化器
        args,# 这里面可以放置一开始就设置好的参数
        epo, # 当前epoch数是第epo轮
        B=0.0,
    ):

    ############### 开始训练 ###################
    T,d,d1 = 0.0,0.0,0.0
    net.train()
    for it, ( Rgb, Thermal, labels, Edge1,Edge0) in enumerate(train_loader):


        Rgb     = Variable(Rgb).cuda()
        Thermal = Variable(Thermal).cuda()
        labels = Variable(labels).cuda().long()
        Edge1 = Variable(Edge1).cuda()
        Edge0 = Variable(Edge0).cuda().long()

        seg_out,edge = net(Rgb,Thermal)

        dual_task = DualTaskLoss()
        focal_loss = FocalLoss(9)
        Edge_focal_loss = Edge_FocalLoss(2)
        a_loss = ( focal_loss(seg_out, labels) + Edge_attention(seg_out, labels,Edge1 ) )#分割损失
        b_loss = ( Edge_focal_loss(edge, Edge0)+ Edge_attention(edge,    Edge0, Edge1 ) * 5. )#边界损失
        c_loss =  dual_task(seg_out, labels) * 25.# 边界分割协同损失
        Trainloss = a_loss + b_loss + c_loss

        #2.2 back propagation
        optimizer.zero_grad()  # reset gradient# 清空梯度
        Trainloss.backward()# 梯度累加
        optimizer.step()  # update parameters of net# 更新参数
        d_loss = F.cross_entropy(seg_out, labels)
        d += Trainloss.data.cpu().numpy()
        d1 += d_loss.data.cpu().numpy()

        ################ 显示损失值与可视化结果 #################
        args.Traintimes += 1
        Trainloss = Trainloss.data.cpu().numpy()
        viz.line([Trainloss], [args.Traintimes], win='TrainLoss', update='append')
        for param_group in optimizer.param_groups:
            lr_this_epo = param_group['lr']

        print('epo %s/%s|train iter %s/%s|total_loss: %.4f|'
              'a_loss: %.4f|b_loss: %.4f|c_loss: %.4f|d_loss: %.4f,'
              '当前学习率：%.4f' \
            % (epo+1, args.e_max, it+1,len(train_dataset)//args.batch_size,float(Trainloss),
               float(a_loss),float(b_loss), float(c_loss),float(d_loss),
               lr_this_epo))

    return  d,d1

def Tra_Val_Tes(args,viz):

    ################ 训练验证测试数据集设置 #################
    ################ 训练数据集
    train_dataset = MF_dataset0(args.data_dir, 'train_1', have_label='True')
    train_loader  = DataLoader(
        dataset     = train_dataset,
        batch_size  = args.batch_size,
        shuffle     = True,
        num_workers = args.num_workers,
        pin_memory  = True,
        drop_last   = True,
    )

    net_ = net(num_classes=9).cuda()
    optimizer = torch.optim.SGD(net.parameters(), lr=args.lr_start, momentum=0.9, weight_decay=0.0005,nesterov=True)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_decay, last_epoch=-1)

    loss1, loss2, loss3,loss4,lr_this_epo = 0.0,0.0,0.0,0.0,0.02
    for e in range(args.e_max):

        ############## 训练 ####################
        loss1,loss2 = train(
            train_dataset=train_dataset,
            train_loader=train_loader,
            args=args,
            epo=e,
            net=net_,
            optimizer=optimizer,
            viz=viz,
        )
        scheduler.step()

        torch.save(net.state_dict(), 'Model/Net--' + str(e) + '.pth')