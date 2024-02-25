#!bin/bash

export IMAGE_SIZE=64
export CHECKPOINT="/logs/SLATE_pair/checkpoint.pt.tar"
export NOBJECTS=15
export DATA_PATH=/data/procthor

python evaluation/eval_slate.py --image_pair True --batch_size 50 --epochs $EPOCHS --image_size $IMAGE_SIZE  --data_path=$DATA_PATH --checkpoint_path $CHECKPOINT