#!bin/bash

#source etc/setup.sh

export IMAGE_SIZE=64
export EPOCHS=50
#export CHECKPOINT="checkpoint.pt.tar"
#export RUNID="19id4uu4"
export NOBJECTS=15
export DATA_PATH=/home/linyan/data/procthor

python train_slot_attention.py --batch_size 64 --epochs $EPOCHS --image_size $IMAGE_SIZE  --data_path=$DATA_PATH --task procthor --max_num_objects=$NOBJECTS
