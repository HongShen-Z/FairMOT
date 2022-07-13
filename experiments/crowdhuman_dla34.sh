#!/bin/bash
#BSUB -J train_ch
#BSUB -o %J.out
#BSUB -e %J.err
#BSUB -q gpu_v100
#BSUB -gpu "num=1:mode=exclusive_process:aff=yes"

module load anaconda3
source activate
conda deactivate
conda activate fairmot
cd src
python train.py mot --exp_id ch_dla34 --gpus $CUDA_VISIBLE_DEVICES \
--batch_size 48 --load_model '../models/ctdet_coco_dla_2x.pth' --num_epochs 60 \
--lr_step '50' --lr 4e-4 --data_cfg '../src/lib/cfg/crowdhuman.json'
cd ..