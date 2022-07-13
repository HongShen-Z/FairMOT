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
python train.py mot --exp_id mix_ft_ch_dla34 --load_model '../models/crowdhuman_dla34.pth' \
--data_cfg '../src/lib/cfg/data.json' --gpus $CUDA_VISIBLE_DEVICES \
--batch_size 32 --lr 3e-4
cd ..