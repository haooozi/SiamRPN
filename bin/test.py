from __future__ import absolute_import

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from got10k.experiments import *
from siamrpn import SiamRPNTracker


if __name__=='__main__':

    net_path = '../models/siamrpn_46.pth'
    tracker = SiamRPNTracker(net_path)
    root_dir = os.path.abspath('F:/OTB100')
    experiment = ExperimentOTB(root_dir, version='tb100')
    experiment.run(tracker, visualize=False)
    prec_score, succ_score, succ_rate = experiment.report([tracker.name])

    ss = '-prec_score:%.3f -succ_score:%.3f -succ_rate:%.3f' % (float(prec_score), float(succ_score), float(succ_rate))
    print(net_path.split('/')[-1], ss)

    # root_dir = os.path.abspath('datasets/OTB')
    # e = ExperimentOTB(root_dir, version=2013)

    # root_dir = os.path.abspath('datasets/UAV123')
    # e = ExperimentUAV123(root_dir, version='UAV123')

    # root_dir = os.path.abspath('datasets/UAV123')
    # e = ExperimentUAV123(root_dir, version='UAV20L')

    # root_dir = os.path.abspath('datasets/DTB70')
    # e = ExperimentDTB70(root_dir)

    # root_dir = os.path.abspath('datasets/VOT2018')           # VOT测试在评估阶段报错
    # e = ExperimentVOT(root_dir,version=2018,read_image=True, experiments=('supervised', 'unsupervised'))

    # root_dir = os.path.abspath('datasets/TColor128')
    # e = ExperimentTColor128(root_dir)

    # root_dir = os.path.abspath('datasets/Nfs')
    # e = ExperimentNfS(root_dir)

    # root_dir = os.path.abspath('datasets/LaSOT')
    # e = ExperimentLaSOT(root_dir)


