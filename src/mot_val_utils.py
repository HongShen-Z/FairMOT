import os.path as osp
import numpy as np


data_root = '~/datasets/MOT/data/gt/mot_challenge'  # the root directory of the dataset
gt_folder = osp.join(data_root, 'MOT17-train')
seqs_str = '''MOT17-02-SDP
              MOT17-04-SDP
              MOT17-05-SDP
              MOT17-09-SDP
              MOT17-10-SDP
              MOT17-11-SDP
              MOT17-13-SDP'''
seqs = [seq.strip() for seq in seqs_str.split()]


def gen_gt_val():
    for seq in seqs:
        print('start seq {}'.format(seq))
        seq_info = open(osp.join(gt_folder, seq, 'seqinfo.ini')).read()
        seqLength = int(seq_info[seq_info.find('seqLength=') + 10:seq_info.find('\nimWidth')])
        gt_txt = osp.join(gt_folder, seq, 'gt', 'gt.txt')
        gt = np.loadtxt(gt_txt, dtype=np.float64, delimiter=',')
        save_val_gt = osp.join(gt_folder, seq, 'gt', 'gt_val_half.txt')
        val_start = seqLength // 2
        with open(save_val_gt, 'w') as f:
            for i, obj in enumerate(gt):
                label_str = '{:d},{:d},{:d},{:d},{:d},{:d},{:d},{:d},{:f}\n'.format(
                    int(obj[0]), int(obj[1]), int(obj[2]), int(obj[3]), int(obj[4]), int(obj[5]), int(obj[6]),
                    int(obj[7]), obj[8])
                if obj[0] > val_start:
                    f.write(label_str)


if __name__ == '__main__':
    gen_gt_val()
