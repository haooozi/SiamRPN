import torch
import torch.nn.functional as F
from torch import nn
from siamrpn.config import config

class SiamRPNNet(nn.Module):
    def __init__(self, init_weight=False):
        super(SiamRPNNet, self).__init__()
        self.featureExtract = nn.Sequential(
            nn.Conv2d(3, 96, 11, stride=2),  #stride=2  [batch,3,127,127]->[batch,96,59,59]
            nn.BatchNorm2d(96),
            nn.MaxPool2d(3, stride=2),       #stride=2  [batch,96,58,58]->[batch,96,29,29]
            nn.ReLU(inplace=True), 
            nn.Conv2d(96, 256, 5),           #[batch,256,29,29]->[batch,256,25,25]
            nn.BatchNorm2d(256),
            nn.MaxPool2d(3, stride=2),       #stride=2  [batch,256,25,25]->[batch,256,12,12]
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 384, 3),          #[batch,256,12,12]->[batch,384,10,10]
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, 3),          #[batch,384,10,10]->[batch,384,8,8]
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, 3),          #[batch,384,8,8]->[batch,256,6,6]
            nn.BatchNorm2d(256),
        )
        self.anchor_num = config.anchor_num    #每一个位置有5个anchor
        """ 模板的分类和回归"""
        self.examplar_cla = nn.Conv2d(256, 256 * 2 * self.anchor_num, kernel_size=3, stride=1, padding=0)
        self.examplar_reg = nn.Conv2d(256, 256 * 4 * self.anchor_num, kernel_size=3, stride=1, padding=0)
        """ 搜索图像的分类和回归"""
        self.instance_cla = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=0)
        self.instance_reg = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=0)
        #这一步是SiamRPN框架中没有的，1x1的卷积，感觉可有可无，仅仅用于回归分支
        #简单理解就是增加了网络的学习能力
        self.regress_adjust = nn.Conv2d(4 * self.anchor_num, 4 * self.anchor_num, 1)
        if init_weight:
            self._init_weights()
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight, 1) #xavier是参数初始化，它的初始化思想是保持输入和输出方差一致，这样就避免了所有输出值都趋向于0
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)     #偏置初始化为0
            elif isinstance(m, nn.BatchNorm2d):      #在激活函数之前，希望输出值由较好的分布，以便于计算梯度和更新参数，这时用到BatchNorm2d函数
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    """——————————前向传播用于训练——————————————————"""
    def forward(self, template, detection):
        N = template.size(0)    # batch=32
        template_feature = self.featureExtract(template)    #[32,256,6,6]
        detection_feature = self.featureExtract(detection)  #[32,256,24,24]
        """对应模板分支，求分类核以及回归核"""
        # [32,256*2*5,4,4]
        kernel_score = self.examplar_cla(template_feature)
        # 类似可得[32,4*5*256,4,4]
        kernel_regression = self.examplar_reg(template_feature)
        """对应搜素图像的分支，得到搜索图像的特征图"""
        conv_score = self.instance_cla(detection_feature)      #[32,256,22,22]
        conv_regression = self.instance_reg(detection_feature)   #[32,256,22,22]
        """对应模板和搜索图像的分类"""
        # [32,256,22,22]->[1,32*256=8192,22,22]
        conv_scores = conv_score.reshape(1, -1, 22, 22)
        score_filters = kernel_score.reshape(-1, 256, 4, 4)    #[32*2*5,256,4,4]
        #input=[1,8192,22,22],filter=[320,256,4,4],得到output=[1,32*2*5,19,19]->[32,10,19,19] 32始终是batch不变，勿忘！
        pred_score = F.conv2d(conv_scores, score_filters, groups=N).reshape(N, 10, 19,19)
        """对应模板和搜索图像的回归----------"""
        #[32,256,22,22]->[1,32*256=8192,22,22]
        conv_reg = conv_regression.reshape(1, -1, 22, 22)
        reg_filters = kernel_regression.reshape(-1, 256, 4, 4)   #[32*4*5,256,4,4]
        #inout=[1,8192,22,22],filter=[340,256,4,4],得到output=[1,32*4*5,19,19]->[32,4*5=20,19,19]——>微调
        pred_regression = self.regress_adjust(F.conv2d(conv_reg, reg_filters, groups=N).reshape(N, 20, 19, 19))
        #score.shape=[32,10,19,19],regression.shape=[32,20,19,19]
        return pred_score, pred_regression
    """—————————————初始化————————————————————"""
    def track_init(self, template):
        N = template.size(0) #1
        template_feature = self.featureExtract(template)# [1,256, 6, 6]
        # kernel_score=[1,2*5*256,4,4]   kernel_regression=[1,4*5*256,4,4]
        kernel_score = self.examplar_cla(template_feature)
        kernel_regression = self.examplar_reg(template_feature)
        self.score_filters = kernel_score.reshape(-1, 256, 4, 4)    #[2*5,256,4,4]
        self.reg_filters = kernel_regression.reshape(-1, 256, 4, 4) #[4*5,256,4,4]
    """—————————————————跟踪—————————————————————"""
    def track_update(self, detection):
        N = detection.size(0)
        # [1,256,24,24]
        detection_feature = self.featureExtract(detection)
        """----得到搜索图像的feature map-----"""
        conv_score = self.instance_cla(detection_feature)     #[1,256,22,22]
        conv_regression = self.instance_reg(detection_feature)  #[1,256,22,22]
        """---------与模板互相关----------"""
        #input=[1,256,22,22] filter=[2*5,256,4,4] gropu=1 得output=[1,2*5,19,19]
        pred_score = F.conv2d(conv_score, self.score_filters, groups=N)
        # input=[1,256,22,22] filter=[4*5,256,4,4] gropu=1 得output=[1,4*5,19,19]
        pred_regression = self.regress_adjust(F.conv2d(conv_regression, self.reg_filters, groups=N))
        #score.shape=[1,10,19,19],regression.shape=[1,20,19,19]
        return pred_score, pred_regression


if __name__ == '__main__':
    model = SiamRPNNet()
    z_train = torch.randn([32,3,127,127])  #batch=8
    x_train = torch.randn([32,3,271,271])
    # 返回shape为[32,20,19,19] [32,10,19,19]  20=5*4 10=5*2
    pred_score_train, pred_regression_train = model(z_train,x_train)
    z_test = torch.randn([1,3,127,127])
    x_test = torch.randn([1,3,271,271])
    model.track_init(z_test)
    # 返回shape为[1,20,19,19] [1,10,19,19]
    pred_score_test, pred_regression_test = model.track_update(x_test)


