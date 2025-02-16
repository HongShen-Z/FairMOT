from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import logging
import os
import os.path as osp
from opts import opts
from tracking_utils.utils import mkdir_if_missing
from tracking_utils.log import logger
import datasets.dataset.jde as datasets
from track import eval_seq


logger.setLevel(logging.INFO)


def demo(opt):
    result_root = opt.output_root if opt.output_root != '' else '.'
    mkdir_if_missing(result_root)

    logger.info('Starting tracking...')
    # print(opt.gpus, '---------------------')
    dataloader = datasets.LoadVideo(opt.input_video, opt.img_size)
    result_filename = os.path.join(result_root, 'results.txt')
    frame_rate = dataloader.frame_rate

    frame_dir = None if opt.output_format == 'text' else osp.join(result_root, 'frame')
    eval_seq(opt, dataloader, 'mot', result_filename,
             save_dir=frame_dir, show_image=False, frame_rate=frame_rate,
             use_cuda=opt.gpus!=[-1])

    if opt.output_format == 'video':
        output_video_path = osp.join(result_root, 'MOT16-03-results.mp4')
        cmd_str = 'ffmpeg -f image2 -i {}/%05d.jpg -b 5000k -c:v mpeg4 {}'.format(osp.join(result_root, 'frame'), output_video_path)
        os.system(cmd_str)


if __name__ == '__main__':
    opt = opts().init()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(opt.gpus[0])  # '0'
    # import torch
    # from torch.utils.cpp_extension import CUDA_HOME
    #
    # cuda_available = torch.cuda.is_available()
    #
    # is_installation_ok = cuda_available and CUDA_HOME is not None
    #
    # print('Is CUDA OK: {0}\nIs cuda available: {1}\nCuda home: {2}'.format(
    #     is_installation_ok, cuda_available, CUDA_HOME))
    demo(opt)
