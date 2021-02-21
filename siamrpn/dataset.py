# -*- coding: utf-8 -*-
import random
from siamrpn.utils import *
from PIL import Image
from siamrpn.config import config
from torch.utils.data import Dataset
from got10k.datasets import GOT10k
from torchvision.transforms import transforms
class GOT10kDataset(Dataset):
    def __init__(self, seq_dataset, z_transforms, x_transforms, name = 'GOT-10k'):

        self.max_inter     = config.frame_range_got #100取两帧最大得间隔
        self.z_transforms  = z_transforms
        self.x_transforms  = x_transforms
        self.sub_class_dir = seq_dataset
        self.ret           = {}
        self.count         = 0
        self.index         = 3000
        self.name          = name
        """生成anchor,,返回1805个anchor，每个anchor是4维[center_x,center_y,w,h],,anchor.shape=[1805,4],,1805=5*19*19"""
        self.anchors       = generate_anchors( config.total_stride,  #8
                                                    config.anchor_base_size, #8
                                                    config.anchor_scales,  #[8] list 列表
                                                    config.anchor_ratios,  #[0.33, 0.5, 1, 2, 3] list 列表
                                                    config.score_size)  #19

    """
    #随机选择某视频序列间隔大于100的两张图片，并保存以下信息
        1、两帧图片路径 2、两帧图片[l,t,w,h]
        self.ret['template_img_path']      = template_img_path   
        self.ret['detection_img_path']     = detection_img_path
        self.ret['template_target_x1y1wh'] = template_gt 
        self.ret['detection_target_x1y1wh']= detection_gt
        3、两帧图片[center_x,center_y,w,h]
        t1, t2 = self.ret['template_target_x1y1wh'].copy(), self.ret['detection_target_x1y1wh'].copy()
        self.ret['template_target_xywh']   = np.array([t1[0]+t1[2]//2, t1[1]+t1[3]//2, t1[2], t1[3]], np.float32)
        self.ret['detection_target_xywh']  = np.array([t2[0]+t2[2]//2, t2[1]+t2[3]//2, t2[2], t2[3]], np.float32)
        4、锚  anchor.shape=(1805,4) ,1805=5*19*19 每个anchor[l,t,w,h]
        self.ret['anchors'] = self.anchors
    """
    def _pick_img_pairs(self, index_of_subclass):
        #0~9334 < 9335，所以不会报错
        assert index_of_subclass < len(self.sub_class_dir), 'index_of_subclass should less than total classes'
        #第index_of_subclass个视频序列的所有帧路径
        video_name = self.sub_class_dir[index_of_subclass][0]
        #第index_of_subclass个视频序列有多少帧
        video_num  = len(video_name)
        #第index_of_subclass个视频序列的所有帧groundtruth
        video_gt   = self.sub_class_dir[index_of_subclass][1]

        status = True
        while status:
            if self.max_inter >= video_num-1:
                self.max_inter = video_num//2
            #比0小取0，比video_num-1大取video_num-1, 但0~video_num-100之间的数肯定在[0,video_num-1]之间，clip作用确定性作用
            # template_index=100+template_index
            template_index = np.clip(random.choice(range(0, max(1, video_num - self.max_inter))), 0, video_num-1)
            detection_index= np.clip(random.choice(range(1, max(2, self.max_inter))) + template_index, 0, video_num-1)
            #获取两帧图片的路径
            template_img_path, detection_img_path  = video_name[template_index], video_name[detection_index]
            #获取两帧图片的gt [l,t,w,h]
            template_gt  = video_gt[template_index]
            detection_gt = video_gt[detection_index]

            #判断两帧图片的w,h是否有都没0，都没得话 status=False，跳出死循环
            if template_gt[2]*template_gt[3]*detection_gt[2]*detection_gt[3] != 0:
                status = False
            #有为0得话，打印出gt信息
            else:
                print(  'index_of_subclass:', index_of_subclass, '\n',
                        'template_index:', template_index, '\n',
                        'template_gt:', template_gt, '\n',
                        'detection_index:', detection_index, '\n',
                        'detection_gt:', detection_gt, '\n')

        # 把信息装入字典
        self.ret['template_img_path']      = template_img_path
        self.ret['detection_img_path']     = detection_img_path
        self.ret['template_target_x1y1wh'] = template_gt 
        self.ret['detection_target_x1y1wh']= detection_gt
        #再把gt[l,t,w,h]->[center_x,center_y,w,h]装入字典
        t1, t2 = self.ret['template_target_x1y1wh'].copy(), self.ret['detection_target_x1y1wh'].copy()
        self.ret['template_target_xywh']   = np.array([t1[0]+t1[2]//2, t1[1]+t1[3]//2, t1[2], t1[3]], np.float32)
        self.ret['detection_target_xywh']  = np.array([t2[0]+t2[2]//2, t2[1]+t2[3]//2, t2[2], t2[3]], np.float32)
        #anchor得信息也装入字典
        self.ret['anchors'] = self.anchors

    """
        #对选择好的两张图片裁剪，返回一下信息
            1、模板和搜索图像->[127,127,3] [271,271,3]
            self.ret['exemplar_img'] = exemplar_img
            self.ret['instance_img'] = instance_img
            2、目标在搜索图像的宽w和高h还有a_x_和b_y_，这两个数是(-12~12内的随机数)
            为什么是a_x_和b_y_呢，因为anchor是以图片中心点作为为(0,0)，而不是左上角，gt要对应起来
            self.ret['cx, cy, w, h'] = [int(a_x_), int(b_y_), w, h]  
        """
    def open(self):
        #读取模板图片(h,w,3)
        template_img = Image.open(self.ret['template_img_path'])
        template_img = np.array(template_img)
        #读取搜索图像图片(h,w,3)
        detection_img = Image.open(self.ret['detection_img_path'])
        detection_img = np.array(detection_img)
        # 随机数小于<0.25
        if np.random.rand(1) < config.gray_ratio:
            #??????????????这是为何 哦，数据增强的过程，没啥用感觉
            template_img = cv2.cvtColor(template_img, cv2.COLOR_RGB2GRAY)
            template_img = cv2.cvtColor(template_img, cv2.COLOR_GRAY2RGB)
            detection_img = cv2.cvtColor(detection_img, cv2.COLOR_RGB2GRAY)
            detection_img = cv2.cvtColor(detection_img, cv2.COLOR_GRAY2RGB)
        """————————————————————————————-模板图像————————————————————————————"""
        #图像shape为(h,w,3),对0和1维求均值就是图像的平均像素值,img_mean.shape=(3,),分别代表R\G\B的均值
        img_mean = np.mean(template_img, axis=(0, 1))
        # 返回127x127x3的图片，缩放因子127/sqrt([w+(w+h)/2]*[h+(w+h)/2])，
        # 目标边界框sqrt([w+(w+h)/2]*[h+(w+h)/2])，目标在127x127x3图片中的宽和高
        exemplar_img, scale_z, s_z, w_x, h_x = self.get_exemplar_image( template_img, #模板图片
                                    self.ret['template_target_xywh'],   #[center_x,center_y,w,h]
                                    config.exemplar_size,   #127
                                    config.context_amount, img_mean )   #context_amount=0.5

        self.ret['exemplar_img'] = exemplar_img
        """————————————————————————————搜索图像————————————————————————————"""
        #获取目标的中心，宽高
        cx, cy, w, h = self.ret['detection_target_xywh']
        """-------为了得到a_x,a_y，再在crop中数据增强---"""
        wc_z = w + 0.5 * (w + h)
        hc_z = h + 0.5 * (w + h)
        s_z = np.sqrt(wc_z * hc_z)  #sqrt([w+(w+h)/2]*[h+(w+h)/2])
        s_x = s_z / (config.instance_size//2)  # 为什么//2？？  2约等于271/127 本来s_x = s_z*127 / config.instance_size
        img_mean_d = tuple(map(int, detection_img.mean(axis=(0, 1))))
        shift_x = np.random.choice(range(-12,12))
        a_x = shift_x * s_x
        shift_y = np.random.choice(range(-12,12))
        b_y = shift_y * shift_x
        """-------a_x,b_y对应原图像，a_x_,b_x_对应271的图像-----————————————————————————————————-"""
        # 返回裁剪后的搜索图像，数据增强因子a_x、b_y，目标相对271x271大小的宽高、缩放因子 271/min(h,w)
        # 这里的h和w是原始图像得到待resize图像的h和w，即裁剪上下文信息并填充后的h和w
        # 不像模板中的缩放因子127/sqrt([]*[]),因为模板中的高宽都是sqrt([]*[]),
        # 而搜索图像这里对高和宽进行了(-0.15~0.15)的缩放，所以高宽不等，先择min(h,w)来计算
        instance_img, a_x, b_y, w_x, h_x, scale_x = self.get_instance_image(  detection_img, self.ret['detection_target_xywh'],
                                                                    config.exemplar_size,  # 127
                                                                    config.instance_size,  # 255
                                                                    config.context_amount, # 0.5
                                                                    a_x, b_y,
                                                                    img_mean_d )
        size_x = config.instance_size   #271
        xmin, ymin = int((size_x + 1) / 2 - w_x / 2), int((size_x + 1) / 2 - h_x / 2)
        xmax, ymax = int((size_x + 1) / 2 + w_x / 2), int((size_x + 1) / 2 + h_x / 2)
        w  = xmax - xmin
        h  = ymax - ymin
        cx = xmin + w/2
        cy = ymin + h/2

        self.ret['instance_img'] = instance_img
        #？？？？？？？？？？？？为什么是a_x_和b_y_，不是cx和cy
        self.ret['cx, cy, w, h'] = [int(shift_x), int(shift_y), w, h]

    def get_exemplar_image(self, img, bbox, size_z, context_amount, img_mean=None):
        cx, cy, w, h = bbox #[center_x,center_y,w,h]
        wc_z = w + context_amount * (w + h)  #包含上下文信息 w+(w+h)/2
        hc_z = h + context_amount * (w + h)  #h+(w+h)/2
        s_z = np.sqrt(wc_z * hc_z)           #sqrt([w+(w+h)/2]*[h+(w+h)/2])
        scale_z = size_z / s_z               #127/sqrt([w+(w+h)/2]*[h+(w+h)/2])  ???为什么127/
        #返回127x127x3的裁剪图片和 127/shape[0]  包含上下文的目标->127x127x3的宽度缩放(这里是缩小)因子
        exemplar_img, scale_x = self.crop_and_pad_old(img, cx, cy, size_z, s_z, img_mean)
        #计算127x127x3图片的高宽，即将原始图片的bbox映射到127大小图片的bbox
        w_x = w * scale_x
        h_x = h * scale_x
        #返回127x127x3的图片，缩放因子，目标边界框sqrt([w+(w+h)/2]*[h+(w+h)/2])，目标在127x127x3图片中的高宽
        return exemplar_img, scale_z, s_z, w_x, h_x
    #比模板图像裁剪多了两个参数a_x,b_y
    def get_instance_image(self, img, bbox, size_z, size_x, context_amount, a_x, b_y, img_mean=None):
        cx, cy, w, h = bbox   #[center_x,center_y,w,h]
        wc_z = w + context_amount * (w + h)
        hc_z = h + context_amount * (w + h)
        s_z = np.sqrt(wc_z * hc_z)  #模板的的上下文信息
        s_x = s_z * size_x / size_z #搜索图像的上下文信息 271/127 * sqrt([w+(w+h)/2]*[h+(w+h)/2])
        # 返回271x271x3大小的图片，该图片内目标的宽、高，缩放因子、宽、高比例(小的为1)
        instance_img, gt_w, gt_h, scale_x, scale_h, scale_w = self.crop_and_pad(img, cx, cy, w, h, a_x, b_y,  size_x, s_x, img_mean)
        w_x = gt_w
        h_x = gt_h
        #数据增强的位移也要想要乘上比例
        a_x, b_y = a_x*scale_w, b_y*scale_h
        #返回裁剪后的搜索图像，数据增强因子a_x、b_y，目标相对271x271大小的宽高、缩放因子
        return instance_img, a_x, b_y, w_x, h_x, scale_x
    #传入的参数是:图片、目标在图片中心点坐标、目标的宽高、数据增强因子a_x和a_y、model_sz=271、original_sz=271/127*sqrt(xxx),img_mean
    def crop_and_pad(self, img, cx, cy, gt_w, gt_h, a_x, b_y, model_sz, original_sz, img_mean=None):
        im_h, im_w, _ = img.shape
        #数据增强：上下文信息所及缩放一定比例(-0.15~0.15)
        scale_h = 1.0 + np.random.uniform(-0.15, 0.15)
        scale_w = 1.0 + np.random.uniform(-0.15, 0.15)

        xmin = (cx-a_x) - ((original_sz - 1) / 2)* scale_w
        xmax = (cx-a_x) + ((original_sz - 1) / 2)* scale_w
        ymin = (cy-b_y) - ((original_sz - 1) / 2)* scale_h
        ymax = (cy-b_y) + ((original_sz - 1) / 2)* scale_h
        left   = int(self.round_up(max(0., -xmin)))
        top    = int(self.round_up(max(0., -ymin)))
        right  = int(self.round_up(max(0., xmax - im_w + 1)))
        bottom = int(self.round_up(max(0., ymax - im_h + 1)))
        xmin = int(self.round_up(xmin + left))
        xmax = int(self.round_up(xmax + left))
        ymin = int(self.round_up(ymin + top))
        ymax = int(self.round_up(ymax + top))

        r, c, k = img.shape
        if any([top, bottom, left, right]):
            te_im = np.zeros((int((r + top + bottom)), int((c + left + right)), k), np.uint8)
            te_im[top:top + r, left:left + c, :] = img

            if top:
                te_im[0:top, left:left + c, :] = img_mean
            if bottom:
                te_im[r + top:, left:left + c, :] = img_mean
            if left:
                te_im[:, 0:left, :] = img_mean
            if right:
                te_im[:, c + left:, :] = img_mean
            im_patch_original = te_im[int(ymin):int(ymax + 1), int(xmin):int(xmax + 1), :]
        else:
            im_patch_original = img[int(ymin):int((ymax) + 1), int(xmin):int((xmax) + 1), :]

        #求原图的高宽映射到271x271时高宽的缩放因子
        if not np.array_equal(model_sz, original_sz): #正常不等 271不会等于 271/127 * sqrt([]*[])
            h, w, _ = im_patch_original.shape
            #缩放因子scale以短边来计算
            if h < w:
                scale_h_ = 1
                scale_w_ = h/w
                scale = config.instance_size/h
            elif h > w:
                scale_h_ = w/h
                scale_w_ = 1
                scale = config.instance_size/w
            elif h == w:
                scale_h_ = 1
                scale_w_ = 1
                scale = config.instance_size/w

            gt_w = gt_w * scale_w_
            gt_h = gt_h * scale_h_

            gt_w = gt_w * scale
            gt_h = gt_h * scale
            #->[271x271x3]
            im_patch = cv2.resize(im_patch_original, (model_sz, model_sz))
        else:
            im_patch = im_patch_original
        #返回271x271x3大小的图片，该图片内目标的宽、高，缩放因子、宽、高比例(小的为1)
        return im_patch, gt_w, gt_h, scale, scale_h_, scale_w_

    #传入的参数分别为(原始图片，center_x,center_y,127,sqrt([w+(w+h)/2]*[h+(w+h)/2]),img_mean)
    def crop_and_pad_old(self, img, cx, cy, model_sz, original_sz, img_mean=None):
        #图片的高宽
        im_h, im_w, _ = img.shape
        #xmin,xmax,ymin,ymax都是相对于original_sz而言
        xmin = cx - (original_sz - 1) / 2
        xmax = xmin + original_sz - 1
        ymin = cy - (original_sz - 1) / 2
        ymax = ymin + original_sz - 1
        #判断original_sz是否超出图片范围，并得出超出多少
        left = int(self.round_up(max(0., -xmin)))
        top = int(self.round_up(max(0., -ymin)))
        right = int(self.round_up(max(0., xmax - im_w + 1)))
        bottom = int(self.round_up(max(0., ymax - im_h + 1)))
        xmin = int(self.round_up(xmin + left))
        xmax = int(self.round_up(xmax + left))
        ymin = int(self.round_up(ymin + top))
        ymax = int(self.round_up(ymax + top))
        #均值填充
        r, c, k = img.shape
        if any([top, bottom, left, right]):  #只要[top, bottom, left, right]有非0，返回True
            te_im = np.zeros((r + top + bottom, c + left + right, k), np.uint8)
            te_im[top:top + r, left:left + c, :] = img
            if top:
                te_im[0:top, left:left + c, :] = img_mean
            if bottom:
                te_im[r + top:, left:left + c, :] = img_mean
            if left:
                te_im[:, 0:left, :] = img_mean
            if right:
                te_im[:, c + left:, :] = img_mean
            im_patch_original = te_im[int(ymin):int(ymax + 1), int(xmin):int(xmax + 1), :]
        else:
            im_patch_original = img[int(ymin):int(ymax + 1), int(xmin):int(xmax + 1), :]
        if not np.array_equal(model_sz, original_sz):
            #判断model_sz(127)和original_sz(sqrt([w+(w+h)/2]*[h+(w+h)/2]))大小以及元素是否相等，显然不等
            im_patch = cv2.resize(im_patch_original, (model_sz, model_sz))
        else:
            im_patch = im_patch_original
        scale = model_sz / im_patch_original.shape[0]
        #返回127x127x3的裁剪图片和 127/shape[0]  包含上下文的目标->127x127x3的宽度缩放(这里是缩小)因子
        return im_patch, scale

    def round_up(self, value):
        return round(value + 1e-6 + 1000) - 1000

    # 返回regression_target.shape=(1805,4),4:(dx,dy,dw,dh)以及根据iou返回label.shape=(1805,1)
    def _target(self):

        regression_target, conf_target = self.compute_target(self.anchors,
                                                             np.array(list(map(round, self.ret['cx, cy, w, h']))))

        return regression_target, conf_target

    def compute_target(self, anchors, box):
        #返回regression_target.shape=(1805,4),4:(dx,dy,dw,dh)
        regression_target = self.get_target_reg(anchors, box)
        #返回iou.shape=(1805,1)
        iou = self.compute_iou(anchors, box).flatten()
        pos_index = np.where(iou > config.pos_threshold)[0]  #>0.6的为正样本
        neg_index = np.where(iou < config.neg_threshold)[0]  #<0.3的为负样本
        #正样本为1，负样本为0，抛弃的为-1(iou介于0.3到0.6之间)
        label = np.ones_like(iou) * -1
        label[pos_index] = 1
        label[neg_index] = 0
        return regression_target, label
    #返回regression_target.shape=(1805,4),4:(dx,dy,dw,dh)
    def get_target_reg(self, anchors, gt_box):
        #anchor.shape=[1805,4] ge_box.shape(4,)
        anchor_xctr = anchors[:, :1]
        anchor_yctr = anchors[:, 1:2]
        anchor_w = anchors[:, 2:3]
        anchor_h = anchors[:, 3:]
        gt_cx, gt_cy, gt_w, gt_h = gt_box
        #真实gt以及anchor的回归参数，而不是预测pre以及anchor的回归参数
        target_x = (gt_cx - anchor_xctr) / anchor_w  # (1805,1) dx=(gx-Ax)/Aw A:anchor g:groundtruth
        target_y = (gt_cy - anchor_yctr) / anchor_h  # (1805,1) dy
        target_w = np.log(gt_w / anchor_w)  # (1805,1) pw
        target_h = np.log(gt_h / anchor_h)  # (1805,1) ph
        regression_target = np.hstack((target_x, target_y, target_w, target_h)) #(1805,4)
        return regression_target

    def compute_iou(self, anchors, box):
        # anchor.shape=[1805,4],box=[4,]
        if np.array(anchors).ndim == 1:
            anchors = np.array(anchors)[None, :]
        else:
            anchors = np.array(anchors)
        if np.array(box).ndim == 1:
            box = np.array(box)[None, :]
        else:
            box = np.array(box)
        #gt_box.shape=(1805,4) 每个[i,4](i=0...1804)都相同
        gt_box = np.tile(box.reshape(1, -1), (anchors.shape[0], 1))

        anchor_x1 = anchors[:, :1] - anchors[:, 2:3] / 2 + 0.5 #xmin
        anchor_x2 = anchors[:, :1] + anchors[:, 2:3] / 2 - 0.5 #xmax
        anchor_y1 = anchors[:, 1:2] - anchors[:, 3:] / 2 + 0.5 #ymin
        anchor_y2 = anchors[:, 1:2] + anchors[:, 3:] / 2 - 0.5 #ymax
        gt_x1 = gt_box[:, :1] - gt_box[:, 2:3] / 2 + 0.5 #xmin
        gt_x2 = gt_box[:, :1] + gt_box[:, 2:3] / 2 - 0.5 #xmax
        gt_y1 = gt_box[:, 1:2] - gt_box[:, 3:] / 2 + 0.5 #ymin
        gt_y2 = gt_box[:, 1:2] + gt_box[:, 3:] / 2 - 0.5 #ymax

        xx1 = np.max([anchor_x1, gt_x1], axis=0)
        xx2 = np.min([anchor_x2, gt_x2], axis=0)
        yy1 = np.max([anchor_y1, gt_y1], axis=0)
        yy2 = np.min([anchor_y2, gt_y2], axis=0)

        inter_area = np.max([xx2 - xx1, np.zeros(xx1.shape)], axis=0) * np.max([yy2 - yy1, np.zeros(xx1.shape)],axis=0)
        area_anchor = (anchor_x2 - anchor_x1) * (anchor_y2 - anchor_y1)
        area_gt = (gt_x2 - gt_x1) * (gt_y2 - gt_y1)
        iou = inter_area / (area_anchor + area_gt - inter_area + 1e-6)
        #返回iou.shape=(1805,1)
        return iou
    """
    #对裁剪后271、127的图像作转换(数据增强)，并保存信息
        self.ret['train_x_transforms'] = self.x_transforms(self.ret['instance_img'])
        self.ret['train_z_transforms'] = self.z_transforms(self.ret['exemplar_img'])    
    """
    def _tranform(self):
        self.ret['train_x_transforms'] = self.x_transforms(self.ret['instance_img'])
        self.ret['train_z_transforms'] = self.z_transforms(self.ret['exemplar_img'])
    #每个视频序列返回经过处理的模板图像和搜索图像，(1805,4)的regression_target和(1805,1)的label
    def __getitem__(self, index):
        index = random.choice(range(len(self.sub_class_dir)))
        if self.name == 'GOT-10k':
            if  index == 4418 or index == 8627 or index == 8629 or index == 9057 or index == 9058 or index==7787 or index==5911 \
                or index == 6894 or index == 7753:
                index += 3
        self._pick_img_pairs(index)
        self.open()
        self._tranform()
        regression_target, conf_target = self._target()
        self.count += 1
        return  self.ret['train_z_transforms'], self.ret['train_x_transforms'], regression_target, conf_target.astype(np.int64)

    def __len__(self):
        return 64000    

def cxcywhtoltwh(bboxes):
    if len(np.array(bboxes).shape) == 1:
        bboxes = np.array(bboxes)[None, :]
    else:
        bboxes = np.array(bboxes)
    x1 = bboxes[:, 0:1] + 1 / 2 - bboxes[:, 2:3] / 2
    x2 = bboxes[:, 2:3]
    y1 = bboxes[:, 1:2] + 1 / 2 - bboxes[:, 3:4] / 2
    y2 = bboxes[:, 3:4]
    return np.concatenate([x1, y1, x2, y2], 1)

if __name__ == "__main__":
    train_z_transforms = transforms.Compose([transforms.ToTensor()])
    train_x_transforms = transforms.Compose([transforms.ToTensor()])
    root_dir = 'D:/Dataset/GOT-10k'
    seq_dataset = GOT10k(root_dir, subset='train')
    train_data = GOT10kDataset(seq_dataset,z_transforms=train_z_transforms,x_transforms=train_x_transforms)
    instance_img, gt, _, _, _, _ = train_data.__getitem__(0)

    """寻找目标位置"""
    # import cv2
    # cv2.imwrite('b.png',instance_img)
    # gt = np.array(list(map(round, gt)))
    # print(gt) #前两个值是shift_x和shift_y
    # gt[0:2] = gt[0:2]+135  #得到真正的center_x和center_y的坐标
    # gt = cxcywhtoltwh(gt)
    # gt = gt.squeeze(axis=0)
    # print(gt)
    # import matplotlib.pyplot as plt
    # plt.imshow(instance_img)
    # ax = plt.gca()
    # ax.add_patch(plt.Rectangle((gt[0:2]),gt[2],gt[3],color="red", fill=False, linewidth=1))
    # plt.show()




    """画anchor"""
    anchor = generate_anchors(8,8,[8],[0.33, 0.5, 1, 2, 3],19) #[center_x,center_y,w,h]
    import matplotlib.pyplot as plt
    # import numpy as np
    import cv2

    a = cv2.imread('b.png')
    # a = cv2.cvtColor(a,cv2.COLOR_BGR2RGB)
    plt.imshow(a)
    ax = plt.gca()
    aaaaa = anchor+np.array([135,135,0,0])  ##[center_x,center_y,w,h]
    aaaaa = cxcywhtoltwh(aaaaa)

    for i in [0,361,722,1083,1444,222,568,395]:
        ax.add_patch(plt.Rectangle((aaaaa[i][0],aaaaa[i][1]), aaaaa[i][2], aaaaa[i][3], color="red", fill=False, linewidth=1))
    plt.show()

