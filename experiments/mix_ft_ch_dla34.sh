#!/bin/bash
#BSUB -J train_mix
#BSUB -o %J.out
#BSUB -e %J.err
#BSUB -q gpu_v100
#BSUB -gpu "num=2:mode=exclusive_process:aff=yes"

module load anaconda3
source activate
conda deactivate
conda activate fairmot
cd src
python train.py mot --exp_id mix_ft_ch_dla34 --load_model '../exp/mot/crowdhuman_dla34/model_last.pth' \
--data_cfg '../src/lib/cfg/data.json' --gpus $CUDA_VISIBLE_DEVICES \
--batch_size 32 --lr 2e-4
cd ..