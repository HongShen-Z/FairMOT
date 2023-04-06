#!/bin/bash
#BSUB -J mot20_ft
#BSUB -o %J.out
#BSUB -e %J.err
#BSUB -q gpu_v100
#BSUB -gpu "num=2:mode=exclusive_process:aff=yes"

module load anaconda3
source activate
conda deactivate
conda activate fair
cd src
python train.py mot --exp_id mot20_ft_mix_ITP-MMD --load_model '../exp/mix_ft_ch_ITP-MMD-MOT20/models/model_last.pth' \
--data_cfg '../src/lib/cfg/mot20.json' --gpus $CUDA_VISIBLE_DEVICES --multi_loss 'fix' --alpha 0.25 \
--batch_size 24 --lr 1e-4 --hm_shape 'oval' --num_epochs 20 --lr_step '15'
cd ..
