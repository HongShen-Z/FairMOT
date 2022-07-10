#!/bin/bash
#BSUB -J track
#BSUB -o %J.out
#BSUB -e %J.err
#BSUB -q gpu_v100
#BSUB -gpu "num=1:mode=exclusive_process:aff=yes"

module load anaconda3
source activate
conda deactivate
conda activate fairmot
python demo.py mot --load_model ../models/fairmot_dla34.pth --conf_thres 0.4 --gpus $CUDA_VISIBLE_DEVICES

#~/.cache/torch/checkpoints/yolov5s.pt
