#!/bin/bash
#BSUB -J arch-dev1
#BSUB -o %J.out
#BSUB -e %J.err
#BSUB -q gpu_v100
#BSUB -gpu "num=1:mode=exclusive_process:aff=yes"

module load anaconda3
source activate
conda deactivate
conda activate fair
cd src
python train.py mot --exp_id abl_fo_arch-dev1 --num_epochs 30 --lr_step '20' --multi_loss 'fix' \
--gpus $CUDA_VISIBLE_DEVICES --load_model '../models/ctdet_coco_dla_2x.pth' --hm_shape 'oval' \
--data_cfg 'lib/cfg/data_half.json' --batch_size 12 --lr 1e-4
cd ..