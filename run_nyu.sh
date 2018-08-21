#!/bin/bash

####### data information ####
ROOT_DIR="/home/gxdai/MMVC_LARGE2/Guoxian_Dai/data/cub_2011/CUB_200_2011/images"
IMAGE_TXT="/home/gxdai/MMVC_LARGE2/Guoxian_Dai/data/cub_2011/CUB_200_2011/images.txt"
TRAIN_TEST_SPLIT_TXT="/home/gxdai/MMVC_LARGE2/Guoxian_Dai/data/cub_2011/CUB_200_2011/train_test_split.txt"
LABEL_TXT="/home/gxdai/MMVC_LARGE2/Guoxian_Dai/data/cub_2011/CUB_200_2011/image_class_labels.txt"





GPU_ID=$1
MODE=$2
CUDA_VISIBLE_DEVICES=$GPU_ID python main.py \
                        --mode $MODE \
                        --optimizer momentum \
                        --batch_size 64 \
                        --learning_rate 0.001 \
                        --learning_rate_decay_type fixed \
                        --loss_type contrastive_loss \
                        --margin 5. \
                        --root_dir $ROOT_DIR \
                        --image_txt $IMAGE_TXT \
                        --train_test_split_txt $TRAIN_TEST_SPLIT_TXT \
                        --label_txt $LABEL_TXT

if [ 0 -eq 1 ]; then
	--num_epochs1 200 \
	--batch_size 64 \
	--restore_ckpt 1 \
	--evaluation 1 \
	--ckpt_dir "./models/scratch/momentumOptimizer" \
	--dn_train 1 \
	--dn_test 1 \
	--weightFile "./models/scratch/momentumOptimizer/models-5" \
	--targetNum 110000
fi
# --weightFile "./models/old/my_model.ckpt-57" \
