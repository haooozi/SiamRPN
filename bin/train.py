import torch
import torchvision.transforms as transforms
import numpy as np
import os
from torch.utils.data import DataLoader
from tqdm import tqdm
import sys
sys.path.append(os.getcwd())
from siamrpn.config import config
from siamrpn.network import SiamRPNNet
from got10k.datasets import  GOT10k
from siamrpn.dataset import GOT10kDataset
from siamrpn.loss import rpn_smoothL1, rpn_cross_entropy_balance
from siamrpn.utils import adjust_learning_rate
from torchvision.transforms import transforms
from siamrpn.transforms import ToTensor
# [transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
torch.manual_seed(config.seed)
torch.cuda.manual_seed(config.seed)
def train(data_dir, net_path=None):
    seq_dataset_train= GOT10k(data_dir, subset='train')
    """定义数据增强(图像预处理)：归一化、转化为Tensor"""
    train_z_transforms = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize(mean=(0.485, 0.456, 0.406),std=(0.229, 0.224, 0.225))])
    train_x_transforms = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize(mean=(0.485, 0.456, 0.406),std=(0.229, 0.224, 0.225))])
    """建立训练数据"""
    train_dataset  = GOT10kDataset(seq_dataset_train, train_z_transforms, train_x_transforms)
    anchors = train_dataset.anchors  #(1805,4)
    """加载训练数据"""
    trainloader = DataLoader(dataset=train_dataset,
                             batch_size=config.train_batch_size,
                             shuffle=True, num_workers=config.train_num_workers,
                             pin_memory=True, drop_last=True)
    """"————————————开始训练——————————————————————"""
    model = SiamRPNNet()
    model = model.cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=config.lr,momentum=config.momentum, weight_decay=config.weight_decay)
    start_epoch = 1
    #接着训练
    if net_path:
        print("loading checkpoint %s" % net_path + '\n')
        print('------------------------------------------------------------------------------------------------ \n')
        checkpoint = torch.load(net_path)
        if 'epoch' in checkpoint:
            start_epoch = checkpoint['epoch'] + 1
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
        else:
            model.load_state_dict(checkpoint)
        del checkpoint
        torch.cuda.empty_cache()  #缓存清零
        print("loaded checkpoint")

    for epoch in range(start_epoch, config.epoch + 1):
        train_loss = []
        model.train() #开启训练模式

        loss_temp_cls = 0  #分类损失
        loss_temp_reg = 0  #回归损失
        for i, data in enumerate(tqdm(trainloader)):
            #得到模板、搜索图像、目标回归参数、目标分类分数，大小为torch.Size([16,3,127,127])[16,3,271,271][16,1805,4][16,1805]
            exemplar_imgs, instance_imgs, regression_target, conf_target = data
            regression_target, conf_target = regression_target.cuda(), conf_target.cuda()
            """---将模板和搜索图像输入net，得到回归参数和分类分数---"""
            #score.shape=[8,10,19,19],regression.shape=[8,20,19,19]
            pred_score, pred_regression = model(exemplar_imgs.cuda(), instance_imgs.cuda())
            #pre_conf.shape=(8,1805,2)
            pred_conf = pred_score.reshape(-1, 2, config.anchor_num * config.score_size * config.score_size).permute(0,2,1)
            #pred_offset.shape=[16,1805,4]
            pred_offset = pred_regression.reshape(-1, 4,config.anchor_num * config.score_size * config.score_size).permute(0,2,1)
            """——————————————计算分类和回归损失————————————————————-"""
            cls_loss = rpn_cross_entropy_balance(pred_conf, conf_target, config.num_pos, config.num_neg, anchors,
                                                 nms_pos=config.nms_pos, nms_neg=config.nms_neg)
            reg_loss = rpn_smoothL1(pred_offset, regression_target, conf_target, config.num_pos, nms_reg=config.nms_reg)
            loss = cls_loss + config.loss_weight * reg_loss    #分类权重和回归权重 1：5
            """——————————————————————————————————————————————"""
            """--------优化三件套---------------------------"""
            optimizer.zero_grad()
            loss.backward()
            # config.clip=10 ，clip_grad_norm_梯度裁剪，防止梯度爆炸,但我觉得
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip)
            optimizer.step()
            """-----------------------------------------"""
            #当前计算图中分离下来的，但是仍指向原变量的存放位置,requires_grad=false
            train_loss.append(loss.detach().cpu())
            #获取tensor的值有两种方法，转化为.cpu().numpy()或.item()
            loss_temp_cls += cls_loss.detach().cpu().numpy()
            loss_temp_reg += reg_loss.detach().cpu().numpy()

            if (i + 1) % config.show_interval == 0:
                print("[epoch %2d][iter %4d] cls_loss: %.4f, reg_loss: %.4f lr: %.2e"
                           % (epoch, i, loss_temp_cls / config.show_interval, loss_temp_reg / config.show_interval,
                              optimizer.param_groups[0]['lr']))
                loss_temp_cls = 0
                loss_temp_reg = 0
        train_loss = np.mean(train_loss)
        print("EPOCH %d  train_loss: %.4f" % (epoch,train_loss))
        #再保存模型之前调整optimizer的学习率，模型保存时是保存下一次训练的学习率以便接着训练
        adjust_learning_rate(optimizer,config.gamma)
        if epoch % config.save_interval == 0:
            if not os.path.exists('./pretrained/'):
                os.makedirs("./pretrained/")
            save_name = "./pretrained/siamrpn_{}.pth".format(epoch)

            if torch.cuda.device_count() > 1: # 多GPU训练
                new_state_dict=model.module.state_dict()
            else:  #单GPU训练
                new_state_dict=model.state_dict()
            torch.save({
                'epoch': epoch,
                'model': new_state_dict,
                'optimizer': optimizer.state_dict(),
            }, save_name)
            print('save model: {}'.format(save_name))


os.environ["CUDA_VISIBLE_DEVICES"] = "0" #多卡情况下默认多卡训练,如果想单卡训练,设置为"0"

if __name__ == '__main__':
    train('D:/Dataset/GOT-10k')
