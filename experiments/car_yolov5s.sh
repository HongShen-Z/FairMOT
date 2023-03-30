#!/bin/bash
#BSUB -J car
#BSUB -o %J.out
#BSUB -e %J.err
#BSUB -q gpu_v100
#BSUB -gpu "num=1:mode=exclusive_process:aff=yes"

module load anaconda3
source activate
conda deactivate
conda activate fair

cd src
python train.py mot --exp_id car_yolov5s --data_cfg '../src/lib/cfg/car.json' --lr 5e-4 --batch_size 16 \
--wh_weight 0.5 --multi_loss 'fix' --arch 'yolo' --reid_dim 64 --hm_shape 'oval' --gpus $CUDA_VISIBLE_DEVICES \
--output-root '../demos' --num_epochs 30 --lr_step '20' --num_classes 3
cd ..