import torch
import torch.nn
import torch.nn.functional as F
import random
import numpy as np
from siamrpn.utils import nms



# input(batch,1805,2) target(batch,1805) num_pos=16 num_neg=48 anchors(1805,4)
# ohem_pos和ohem_neg都是False，即训练时没有对anchor采用非极大抑制比
def rpn_cross_entropy_balance(input, target, num_pos, num_neg, anchors, nms_pos=None, nms_neg=None):
    loss_all = []
    for batch_id in range(target.shape[0]):  #遍历batch，计算损失
        min_pos = min(len(np.where(target[batch_id].cpu() == 1)[0]), num_pos) #num-pos=16，意思target=1的最多取16个
        min_neg = int(min(len(np.where(target[batch_id].cpu() == 1)[0]) * num_neg / num_pos, num_neg)) #target=0最多48个，且保证是target=1个数的三倍
        pos_index = np.where(target[batch_id].cpu() == 1)[0].tolist() #参数列表
        neg_index = np.where(target[batch_id].cpu() == 0)[0].tolist() #参数列表
        if nms_pos: #False
            pass
        else:
            pos_index_random = random.sample(pos_index, min_pos) #随机取min_pos个正样本的索引
            if len(pos_index) > 0:
                # 交叉熵损失
                pos_loss_bid_final = F.cross_entropy(input=input[batch_id][pos_index_random],target=target[batch_id][pos_index_random], reduction='none')
            else:
                pos_loss_bid_final = torch.FloatTensor([0]).cuda()
        if nms_neg:#False
            pass
        else:
            if len(pos_index) > 0:
                neg_index_random = random.sample(neg_index, min_neg)
                #计算损失时，input要是float型，target为long型,reduction为None表示不计算损失平均
                #如果target为1，把input的第二维度当作计算，target为0时，把input的第一位当作计算
                neg_loss_bid_final = F.cross_entropy(input=input[batch_id][neg_index_random],target=target[batch_id][neg_index_random], reduction='none')
            else:
                neg_index_random = random.sample(np.where(target[batch_id].cpu() == 0)[0].tolist(), num_neg)
                neg_loss_bid_final = F.cross_entropy(input=input[batch_id][neg_index_random],target=target[batch_id][neg_index_random], reduction='none')
        #每张图片的损失平均值
        loss_bid = (pos_loss_bid_final.mean() + neg_loss_bid_final.mean()) / 2
        loss_all.append(loss_bid)
    final_loss = torch.stack(loss_all).mean()
    return final_loss


# input(batch,1805,4) target(batch,1805,4) label(16,1805),num_pos=16  ohem=False(不使用nms)
def rpn_smoothL1(input, target, label, num_pos=16, nms_reg=None):
    loss_all = []
    for batch_id in range(target.shape[0]): #遍历每张图片计算回归损失
        min_pos = min(len(np.where(label[batch_id].cpu() == 1)[0]), num_pos) #最多取16个正样本，回归只针对正样本
        if nms_reg:
            pos_index = np.where(label[batch_id].cpu() == 1)[0]
            if len(pos_index) > 0:
                loss_bid = F.smooth_l1_loss(input[batch_id][pos_index], target[batch_id][pos_index], reduction='none')
                #得到损失值得索引，损失小到大排列
                sort_index = torch.argsort(loss_bid.mean(1))
                #从最后取损失值大得用于训练
                loss_bid_nms = loss_bid[sort_index[-num_pos:]]
            else:
                loss_bid_nms = torch.FloatTensor([0]).cuda()[0]
            loss_all.append(loss_bid_nms.mean())
        else:
            pos_index = np.where(label[batch_id].cpu() == 1)[0]   #lebel=1即正样本的索引列表
            pos_index = random.sample(pos_index.tolist(), min_pos)
            if len(pos_index) > 0:
                loss_bid = F.smooth_l1_loss(input[batch_id][pos_index], target[batch_id][pos_index])
            else:
                loss_bid = torch.FloatTensor([0]).cuda()[0]
            loss_all.append(loss_bid.mean())
    final_loss = torch.stack(loss_all).mean()
    return final_loss

if __name__ == '__main__':
    input = np.array([[0.2,0.8],[0.4,0.6],[0.9,0.1]],dtype=float)
    input = torch.from_numpy(input)
    target = np.array([1,1,0],dtype='int64')
    target = torch.from_numpy(target)
    loss = F.cross_entropy(input,target,reduction='none')
    print(loss)