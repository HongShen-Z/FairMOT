import os
import os.path as osp
import glob
import shutil


path = osp.join(os.getcwd(), '../demos/MOT17_ablation_base/data')
for file in glob.glob(path + '/*SDP.txt'):
    new = osp.basename(file).split('SDP')[0]
    shutil.copy(file, osp.join(path, new + 'DPM.txt'))
    shutil.copy(file, osp.join(path, new + 'FRCNN.txt'))

