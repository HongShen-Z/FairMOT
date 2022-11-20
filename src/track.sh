#!/bin/bash
#BSUB -J track&val
#BSUB -o %J.out
#BSUB -e %J.err
#BSUB -q gpu_v100
#BSUB -gpu "num=1:mode=exclusive_process:aff=yes"

module load anaconda3
source activate
conda deactivate
conda activate fair
python track_half.py mot --val_mot17 True --load_model ../exp/abl_fo_mti/models/model_last.pth \
--conf_thres 0.4 --gpus $CUDA_VISIBLE_DEVICES --exp_id MOT17_fo_mti
python TrackEval/scripts/run_mot_challenge.py --TRACKERS_TO_EVAL 'MOT17_fo_mti' --BENCHMARK 'MOT17' \
--METRICS 'HOTA' 'CLEAR' 'Identity' --SKIP_SPLIT_FOL True --USE_PARALLEL True --NUM_PARALLEL_CORES 2 \
--GT_LOC_FORMAT '{gt_folder}/{seq}/gt/gt_val_half.txt' \
--TRACKERS_FOLDER '/seu_share/home/dijunyong/220205723/projects/FairMOT/demos' \
--GT_FOLDER '/seu_share/home/dijunyong/220205723/datasets/MOT/data/gt/mot_challenge/'
#python track.py mot --val_mot20 True --load_model ../exp/mot/ch_dla34_gd0.5/models/model_60.pth --conf_thres 0.3 \
#--gpus $CUDA_VISIBLE_DEVICES --exp_id MOT20_ch_gd0.5
#python track.py mot --val_mot15 True --load_model ../exp/mot/ch_dla34_wh_eiou4/models/model_320.pth \
#--conf_thres 0.6 --gpus $CUDA_VISIBLE_DEVICES --exp_id MOT15_ch_wh_eiou4

#~/.cache/torch/checkpoints/yolov5s.pt
