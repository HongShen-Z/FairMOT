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
#python train.py mot --exp_id ch_dla34_wh7 --gpus $CUDA_VISIBLE_DEVICES --resume \
#--batch_size 32 --load_model '../exp/mot/ch_dla34_wh6/models/model_last.pth' --num_epochs 600 \
#--lr_step '800' --lr 1e-4 --data_cfg '../src/lib/cfg/crowdhuman.json' --dense_wh --wh_weight 5.0
python train.py mot --exp_id ch_dla34_wh_eiou4 --gpus $CUDA_VISIBLE_DEVICES --resume \
--batch_size 32 --load_model '../exp/mot/ch_dla34_wh_eiou2/models/model_155.pth' --num_epochs 300 \
--lr_step '800' --lr 1e-5 --data_cfg '../src/lib/cfg/crowdhuman.json' --dense_wh --wh_weight 5.0
cd ..
#--load_model '../models/ctdet_coco_dla_2x.pth'