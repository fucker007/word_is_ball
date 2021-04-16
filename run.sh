#!/bin/bash
# need conda activate mask , if dont hava, just install from environment.yaml
CUDA_VISIBLE_DEVICES='0'
python main.py --train_id 0001 \
    --classifer train  \
    --data_path ~/DataPublic/train/  \
    --mode_path models \
    --input_size 224 \
    --epochs 5000 \
    --batch_size 32 \
    --workers 12 \
    --lr 0.01 \
    #--pretrain True \
    #--pretrain_epoch 50 \
