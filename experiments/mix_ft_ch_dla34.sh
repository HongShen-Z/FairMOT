#!/bin/bash
#BSUB -J train_mix
#BSUB -o %J.out
#BSUB -e %J.err
#BSUB -q gpu_v100
#BSUB -gpu "num=2:mode=exclusive_process:aff=yes"

module load anaconda3
source activate
conda deactivate
conda activate fair
cd src
python train.py mot --exp_id mix_ft_ch_ITP-MMD --load_model '../exp/ch_fix_ITP-MMD/models/model_last.pth' \
--data_cfg '../src/lib/cfg/data.json' --gpus $CUDA_VISIBLE_DEVICES --multi_loss 'fix' --alpha 0.25 \
--batch_size 24 --lr 2e-4 --hm_shape 'oval'
cd ..