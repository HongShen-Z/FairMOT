#!/bin/bash
#BSUB -J gen_labels
#BSUB -o %J.out
#BSUB -e %J.err
#BSUB -q gpu_v100
#BSUB -gpu "num=1:mode=exclusive_process:aff=yes"

module load anaconda3
source activate
conda deactivate
conda activate fairmot

python gen_labels_16.py
#~/.cache/torch/checkpoints/yolov5s.pt
