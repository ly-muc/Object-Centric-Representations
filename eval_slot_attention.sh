#!bin/bash

#source etc/setup.sh

export IMAGE_SIZE=128
export EPOCHS=150
export CHECKPOINT="/logs/FrameConsistency/checkpoint.pt.tar"

export NOBJECTS=15
export DATA_PATH=/data/procthor
export DEBUG=True

python evaluation/eval_slotattention.py --image_pair True --batch_size 32 --epochs $EPOCHS --image_size $IMAGE_SIZE  --data_path=$DATA_PATH --task procthor --max_num_objects=$NOBJECTS
