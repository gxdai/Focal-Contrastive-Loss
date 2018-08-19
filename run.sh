
#!/bin/bash

# target: 0.8

GPU_ID=$1
MODE=$2
CUDA_VISIBLE_DEVICES=$GPU_ID py_gxdai main.py \
                        --mode $MODE \
                        --optimizer momentum \
                        --batch_size 64 \
                        --learning_rate 0.001 \
                        --learning_rate_decay_type fixed \
                        --loss_type contrastive_loss \
                        --margin 5.

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
