#!/bin/bash
#BSUB -J gen_labels
#BSUB -o %J.out
#BSUB -e %J.err
#BSUB -q normal

module load anaconda3
source activate
conda deactivate
conda activate fairmot

python gen_labels_15.py
python gen_labels_20.py
#~/.cache/torch/checkpoints/yolov5s.pt
