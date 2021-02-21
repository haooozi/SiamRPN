import numpy as np


class Config:
    # dataset related
    exemplar_size = 127
    instance_size = 271
    context_amount = 0.5
    sample_type = 'uniform'


    nms_pos = False #原始都是False 训练的时候没有对anchors非极大值抑制，具体参考loss.py函数rpn_cross_entropy_balance（）
    nms_neg = False #原始都是False 训练的时候没有对anchors非极大值抑制，具体参考loss.py函数rpn_cross_entropy_balance（）
    nms_reg = True #原始都是False 训练的时候没有对anchors非极大值抑制，具体参考loss.py函数rpn_smoothL1（）
    pairs_per_video_per_epoch = 2
    frame_range_got = 100
    train_batch_size = 8
    train_num_workers = 2
    clip = 10

    start_lr = 1e-2
    end_lr   = 1e-5


    epoch = 50

    lr = np.logspace(np.log10(start_lr), np.log10(end_lr), num=epoch)[0]
    #array([1.00000000e-02, 8.68511374e-03, 7.54312006e-03, 6.55128557e-03,
       # 5.68986603e-03, 4.94171336e-03, 4.29193426e-03, 3.72759372e-03,
       # 3.23745754e-03, 2.81176870e-03, 2.44205309e-03, 2.12095089e-03,
       # 1.84206997e-03, 1.59985872e-03, 1.38949549e-03, 1.20679264e-03,
       # 1.04811313e-03, 9.10298178e-04, 7.90604321e-04, 6.86648845e-04,
       # 5.96362332e-04, 5.17947468e-04, 4.49843267e-04, 3.90693994e-04,
       # 3.39322177e-04, 2.94705170e-04, 2.55954792e-04, 2.22299648e-04,
       # 1.93069773e-04, 1.67683294e-04, 1.45634848e-04, 1.26485522e-04,
       # 1.09854114e-04, 9.54095476e-05, 8.28642773e-05, 7.19685673e-05,
       # 6.25055193e-05, 5.42867544e-05, 4.71486636e-05, 4.09491506e-05,
       # 3.55648031e-05, 3.08884360e-05, 2.68269580e-05, 2.32995181e-05,
       # 2.02358965e-05, 1.75751062e-05, 1.52641797e-05, 1.32571137e-05,
       # 1.15139540e-05, 1.00000000e-05])
    #pepoch=50的话 gamma=0.868511373751352
    gamma = np.logspace(np.log10(start_lr), np.log10(end_lr), num=epoch)[1] / \
            np.logspace(np.log10(start_lr), np.log10(end_lr), num=epoch)[0]
                                           # decay rate of LR_Schedular
    step_size = 1                          # step size of  LR_Schedular
    momentum = 0.9                         # momentum of SGD
    weight_decay = 0.0005                  # weight decay of optimizator
    seed = 6666                            # seed to sample training videos
    max_translate =  12                    # max translation of random shift  随机移动
    scale_resize = 0.15                    # scale step of instance image
    total_stride = 8                       # total stride of backbone
    valid_scope = int((instance_size - exemplar_size) / total_stride + 1)#anchor的范围

    anchor_scales = np.array([8,])
    anchor_ratios = np.array([0.5, 0.66, 1, 1.5, 2])
    anchor_num = len(anchor_scales) * len(anchor_ratios)
    anchor_base_size = 8
    pos_threshold = 0.6
    neg_threshold = 0.3
    num_pos = 16
    num_neg = 48
    loss_weight = 5
    save_interval = 1
    show_interval = 100


    # tracking related
    gray_ratio = 0.25
    blur_ratio = 0.15
    score_size = int((instance_size - exemplar_size) / total_stride + 1)  #19
    penalty_k =0.20
    window_influence = 0.40
    lr_box =0.25 #0.30
    min_scale = 0.1
    max_scale = 10

    def update(self, cfg):
        for k, v in cfg.items():
            setattr(self, k, v)
        self.score_size = (self.instance_size - self.exemplar_size) //self.total_stride + 1
        self.valid_scope= self.score_size

config = Config()
