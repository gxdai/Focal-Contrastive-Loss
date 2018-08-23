#!/bin/bash

####### data information ####
if [ $(hostname) = 'dgx1' ];  then
    # running code on the dgx1

    ROOT_DIR="/data/Guoxian_Dai/CUB_200_2011/CUB_200_2011/images"
    IMAGE_TXT="/data/Guoxian_Dai/CUB_200_2011/CUB_200_2011/images.txt"
    TRAIN_TEST_SPLIT_TXT="/data/Guoxian_Dai/CUB_200_2011/CUB_200_2011/train_test_split.txt"
    LABEL_TXT="/data/Guoxian_Dai/CUB_200_2011/CUB_200_2011/image_class_labels.txt"
    LEARNING_RATE=0.0001
    PYTHON=py_gxdai
elif [ $(hostname) = 'nyu' ]; then
   # running code one nyu machine

    ROOT_DIR="/home/gxdai/MMVC_LARGE2/Guoxian_Dai/data/cub_2011/CUB_200_2011/images"
    IMAGE_TXT="/home/gxdai/MMVC_LARGE2/Guoxian_Dai/data/cub_2011/CUB_200_2011/images.txt"
    TRAIN_TEST_SPLIT_TXT="/home/gxdai/MMVC_LARGE2/Guoxian_Dai/data/cub_2011/CUB_200_2011/train_test_split.txt"
    LABEL_TXT="/home/gxdai/MMVC_LARGE2/Guoxian_Dai/data/cub_2011/CUB_200_2011/image_class_labels.txt"
    LEARNING_RATE=0.0001
fi



GPU_ID=$1
MODE=$2
LOSS_TYPE=$3

CUDA_VISIBLE_DEVICES=$GPU_ID $PYTHON main.py \
                        --mode $MODE \
                        --optimizer "momentum" \
                        --batch_size 64 \
                        --momentum 0.9 \
                        --learning_rate $LEARNING_RATE \
                        --learning_rate_decay_type "fixed" \
                        --loss_type $LOSS_TYPE \
                        --margin 1.0 \
                        --root_dir $ROOT_DIR \
                        --image_txt $IMAGE_TXT \
                        --train_test_split_txt $TRAIN_TEST_SPLIT_TXT \
                        --label_txt $LABEL_TXT \
                        --focal_decay_factor "1.0" \
                        --display_step 20 \
                        --eval_step 20
                        #--with_regularizer





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
