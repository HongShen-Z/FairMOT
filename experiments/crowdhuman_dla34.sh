#!/bin/bash
#BSUB -J train_ch
#BSUB -o %J.out
#BSUB -e %J.err
#BSUB -q gpu_v100
#BSUB -gpu "num=2:mode=exclusive_process:aff=yes"

module load anaconda3
source activate
conda deactivate
conda activate fair
cd src
python train.py mot --exp_id ch_dla34_wh2 --gpus $CUDA_VISIBLE_DEVICES \
--batch_size 32 --load_model '../models/ctdet_coco_dla_2x.pth' --num_epochs 80 \
--lr_step '20,50,70' --lr 1e-3 --data_cfg '../src/lib/cfg/crowdhuman.json' --dense_wh --wh_weight 5.0
cd ..