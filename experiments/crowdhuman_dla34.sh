#!/bin/bash
#BSUB -J train_ch
#BSUB -o %J.out
#BSUB -e %J.err
#BSUB -q gpu_v100
#BSUB -gpu "num=1:mode=exclusive_process:aff=yes"

module load anaconda3
source activate
conda deactivate
conda activate fair
cd src
python train.py mot --exp_id ch_ITP-MMD --gpus $CUDA_VISIBLE_DEVICES --multi_loss 'uncertainty' --alpha 0.25 \
--batch_size 12 --load_model '../models/ctdet_coco_dla_2x.pth' --num_epochs 30 --hm_shape 'oval' \
--lr_step '20' --lr 2e-4 --data_cfg '../src/lib/cfg/crowdhuman.json' --output-root '../demos/abl/net'
#python train.py mot --exp_id ch_res34_whe2 --gpus $CUDA_VISIBLE_DEVICES --arch 'resdcn_34' --resume \
#--batch_size 32 --num_epochs 300 --load_model '../exp/mot/ch_res34_whe1/models/model_last.pth' \
#--lr_step '800' --lr 1e-5 --data_cfg '../src/lib/cfg/crowdhuman.json' --dense_wh --wh_weight 5.0
cd ..
#--load_model '../models/ctdet_coco_dla_2x.pth' '../exp/mot/ch_dla34_wh_eiou3/models/model_210.pth'  --resume