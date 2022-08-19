#!/bin/bash
#BSUB -J track
#BSUB -o %J.out
#BSUB -e %J.err
#BSUB -q gpu_v100
#BSUB -gpu "num=1:mode=exclusive_process:aff=yes"

module load anaconda3
source activate
conda deactivate
conda activate fair
python track.py mot --val_mot20 True --load_model ../exp/mot/ch_dla34_wh5/models/model_last.pth --conf_thres 0.3 \
--gpus $CUDA_VISIBLE_DEVICES --exp_id MOT20_ch_wh5
python track.py mot --val_mot15 True --load_model ../exp/mot/ch_dla34_wh5/models/model_last.pth --conf_thres 0.6 \
--gpus $CUDA_VISIBLE_DEVICES --exp_id MOT15_ch_wh5
#python track.py mot --exp_id MOT15val_dla34 --load_model ../models/fairmot_dla34.pth \
#--conf_thres 0.6 --gpus $CUDA_VISIBLE_DEVICES

#~/.cache/torch/checkpoints/yolov5s.pt
