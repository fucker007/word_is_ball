#!/bin/bash
# need conda activate mask , if dont hava, just install from environment.yaml
CUDA_VISIBLE_DEVICES='0'
python main.py --train_id 0001 \
    --classifer train  \
    --data_path /home/liuyang/workspace/data/classif__fish_11_原件勿动/label/  \
    --val_path /home/liuyang/workspace/data/classif__fish_11_原件勿动/label/validation.txt \
    --mode_path models \
    --input_size 224 \
    --epochs 5000 \
    --batch_size 64 \
    --workers 12 \
    --lr 0.001 \
    --num_class 9 
    #--pretrain True \
    #--pretrain_epoch 50 \
