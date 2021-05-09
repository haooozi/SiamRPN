import numpy as np
import cv2
import torch
import torch.nn.functional as F
import time
import torchvision.transforms as transforms
from siamrpn.network import SiamRPNNet
from siamrpn.config import config
from torchvision.transforms import transforms
from siamrpn.utils import generate_anchors, get_exemplar_image, get_instance_image, box_transform_inv,show_image

#设置用于cpu操作的OpenMP线程数，防止太大占领CPU
torch.set_num_threads(1)

class SiamRPNTracker():
    def __init__(self, model_path):
        self.name='SiamRPN'
        #加载训练好的模型
        self.model = SiamRPNNet()
        checkpoint = torch.load(model_path)
        if 'model' in checkpoint.keys():
            self.model.load_state_dict(torch.load(model_path)['model'])
        else:
            self.model.load_state_dict(torch.load(model_path))
        self.model = self.model.cuda()
        #初始化为eval()，不计算梯度
        self.model.eval()
        #定义转换，仅仅转为Tensor张量
        self.transforms = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=(0.485, 0.456, 0.406),std=(0.229, 0.224, 0.225))])
        #这里生成的anchor与训练一样  (1805,4)
        self.anchors = generate_anchors(config.total_stride, config.anchor_base_size, config.anchor_scales,
                                        config.anchor_ratios,
                                        config.valid_scope)
        #汉宁窗口，可以尝试用余弦窗口
        #[config.anchor_num, 1, 1]对应维度，第一个维度砌砖5倍，第2/3维度不变 最后(5,19,19)->(1805,)
        self.window = np.tile(np.outer(np.hanning(config.score_size), np.hanning(config.score_size))[None, :],
                              [config.anchor_num, 1, 1]).flatten()

    #初始帧
    def init(self, frame, bbox):
        #[l,t,w,h]->[center_x,center_y,w,h]
        self.bbox = np.array([bbox[0]-1 + (bbox[2]-1) / 2 , bbox[1]-1 + (bbox[3]-1) / 2 , bbox[2], bbox[3]])
        #获取目标中心点[c_x,c_y]以便后续使用
        self.pos = np.array([bbox[0]-1 + (bbox[2]-1) / 2 , bbox[1]-1 + (bbox[3]-1) / 2])
        #获取目标宽高[w,h]以便后续使用
        self.target_sz = np.array([bbox[2], bbox[3]])
        #获取目标宽高[w,h]以便后续使用
        self.origin_target_sz = np.array([bbox[2], bbox[3]])
        #获取需要图像R/G/B均值   img_mean.shape=(1,1,3)
        self.img_mean = np.mean(frame, axis=(0, 1))
        #获取模板图像
        #返回127x127x3大小的图像、127/上下文信息
        exemplar_img, scale_z, _ = get_exemplar_image(frame, self.bbox,config.exemplar_size, config.context_amount, self.img_mean)
        #增加一个batch维度再输入网络
        exemplar_img = self.transforms(exemplar_img)[None, :, :, :]
        self.model.track_init(exemplar_img.cuda())

    def update(self, frame):

        #传入上一帧的bbox，以及保持初始帧的img_mean不变来填充后续所有帧
        #返回271x271x3大小的图片、缩放因子是271/(上下文信息x271/127)
        instance_img_np, _, _, scale_x = get_instance_image(frame, self.bbox, config.exemplar_size,
                                                         config.instance_size,
                                                         config.context_amount, self.img_mean)
        """----------得到回归参数并对anchor回归————————————"""
        #增加一个batch维度再送入网络，返回score.shape=[1,10,19,19],regression.shape=[1,20,19,19]
        instance_img = self.transforms(instance_img_np)[None, :, :, :]
        pred_score, pred_regression = self.model.track_update(instance_img.cuda())
        #[1,10,19,19]->[1,2,5*19*19]->[1,1805,2]
        pred_conf = pred_score.reshape(-1, 2, config.anchor_num * config.score_size * config.score_size).permute(0,2,1)
        #[1,20,19,19]->[1,4,5*19*19]->[1,1805,4]
        pred_offset = pred_regression.reshape(-1, 4,config.anchor_num * config.score_size * config.score_size).permute(0,2,1)

        # 使用detach()函数来切断一些分支的反向传播;返回一个新的Variable，从当前计算图中分离下来的，
        # 但是仍指向原变量的存放位置,不同之处只是requires_grad为false，得到的这个Variable永远不需要计算其梯度，不具有grad。
        # 即使之后重新将它的requires_grad置为true,它也不会具有梯度grad #这样我们就会继续使用这个新的Variable进行计算，
        # 后面当我们进行反向传播时，到该调用detach()的Variable就会停止，不能再继续向前进行传播
        delta = pred_offset[0].cpu().detach().numpy()

        #传入的anchor(1805,4) delta(1805,4),delta是回归参数，对anchor进行调整，返回调整后的anchor，即pre_box(1805,4)
        box_pred = box_transform_inv(self.anchors, delta)
        #pred_conf=[1,1805,2]
        #score_pred.shape=torch.Size([1805]) 取1，表示取正样本
        score_pred = F.softmax(pred_conf, dim=2)[0, :, 1].cpu().detach().numpy()#计算预测分类得分
        """--------------------------------------------------"""

        def change(r): 
            return np.maximum(r, 1. / r)

        def sz(w, h):  #返回上下文信息
            pad = (w + h) * 0.5
            sz2 = (w + pad) * (h + pad)
            return np.sqrt(sz2)

        def sz_wh(wh): #返回上下文信息
            pad = (wh[0] + wh[1]) * 0.5
            sz2 = (wh[0] + pad) * (wh[1] + pad)
            return np.sqrt(sz2)
        """---为了得到最终最大分类分数值的索引-------"""
        #尺度惩罚 一个>1的数
        s_c = change(sz(box_pred[:, 2], box_pred[:, 3]) / (sz_wh(self.target_sz * scale_x)))
        #比例惩罚 一个>1的数
        r_c = change((self.target_sz[0] / self.target_sz[1]) / (box_pred[:, 2] / box_pred[:, 3]))
        # 尺度惩罚和比例惩罚 penalty_k=0.22,penalty最大为1，即不惩罚
        penalty = np.exp(-(r_c * s_c - 1.) * config.penalty_k)
        pscore = penalty * score_pred#对每一个anchors的正样本分类预测分数×惩罚因子
        pscore = pscore * (1 - config.window_influence) + self.window * config.window_influence #再乘以余弦窗
        best_pscore_id = np.argmax(pscore) #返回最大得分的索引id
        """-------------------------------------"""

        """---------得到新的目标状态信息用于更新-----------------------------"""
        #这个得到的target[c_x,c_y,w,h]，其中c_x、c_y是以上一帧的pos为(0,0)的坐标(或者说相对偏移)
        target = box_pred[best_pscore_id, :] / scale_x
        # 预测框的学习率  lr_box=0.30,在后续更新w和h时考虑前一帧的影响(类似SGD里的动量)
        lr = penalty[best_pscore_id] * score_pred[best_pscore_id] * config.lr_box
        #关于clip的用法，clip(a,a_min,a_max) a中<a_min的都设为a_min,>a_max都设为a_max
        #将res_x即c_x固定在(0~w)之间、 将res_y即c_y固定在(0~h)之间
        res_x = np.clip(target[0] + self.pos[0], 0, frame.shape[1])
        res_y = np.clip(target[1] + self.pos[1], 0, frame.shape[0])
        #将res_w即w固定在(0.1w~10w)之间、 将res_h即h固定在(0.1h~10h)之间
        #res_w和res_h保留前一帧的宽高，保留成都1-(0.3*惩罚*分数值)
        res_w = np.clip(self.target_sz[0] * (1 - lr) + target[2] * lr, config.min_scale * self.origin_target_sz[0],
                        config.max_scale * self.origin_target_sz[0])
        res_h = np.clip(self.target_sz[1] * (1 - lr) + target[3] * lr, config.min_scale * self.origin_target_sz[1],
                        config.max_scale * self.origin_target_sz[1])
        # res_w = np.clip( target[2] , config.min_scale * self.origin_target_sz[0],
        #                 config.max_scale * self.origin_target_sz[0])
        # res_h = np.clip( target[3] , config.min_scale * self.origin_target_sz[1],
        #                 config.max_scale * self.origin_target_sz[1])
        """---------------------------------------------------------"""

        """-------------更新目标信息：中心、宽、高------------"""
        #更新目标中心点坐标、更新目标宽高
        self.pos = np.array([res_x, res_y])
        self.target_sz = np.array([res_w, res_h])
        bbox = np.array([res_x, res_y, res_w, res_h])

        #将目标约束在图片内
        self.bbox = (
            np.clip(bbox[0], 0, frame.shape[1]).astype(np.float64),
            np.clip(bbox[1], 0, frame.shape[0]).astype(np.float64),
            np.clip(bbox[2], 10, frame.shape[1]).astype(np.float64),
            np.clip(bbox[3], 10, frame.shape[0]).astype(np.float64))
        #[c_x,c_y,w,h]->[l,t,w,h]
        bbox=np.array([# tr-x,tr-y w,h                                  
            self.pos[0] + 1 - (self.target_sz[0]-1) / 2,
            self.pos[1] + 1 - (self.target_sz[1]-1) / 2,
            self.target_sz[0], self.target_sz[1]])
        """-----------------------------------------------------"""
        return bbox

    #传入的是测试数据集(OTB等)每个视频序列的所有帧图片、第一帧标注框
    def track(self, img_files, box, visualize=False):
        frame_num = len(img_files)
        boxes = np.zeros((frame_num, 4))
        boxes[0] = box
        times = np.zeros(frame_num)
        for f, img_file in enumerate(img_files):
            img = cv2.imread(img_file, cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            begin = time.time()
            if f == 0:
                self.init(img, box)
            else:
                boxes[f, :] = self.update(img)
            times[f] = time.time() - begin

            if visualize:
                show_image(img, boxes[f, :])
        return boxes, times
