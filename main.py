from __future__ import print_function

import argparse
import os
import sys

import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.nets
import time
import random
import numpy as np



tf.set_random_seed(22222)

# from utils import *
from datetime import datetime
import model

parser = argparse.ArgumentParser()

parser.add_argument('--root_dir', type=str, \
        default='/data/Guoxian_Dai/CUB_200_2011/CUB_200_2011/images')
parser.add_argument('--image_txt', type=str, \
        default='/data/Guoxian_Dai/CUB_200_2011/CUB_200_2011/images.txt')
parser.add_argument('--train_test_split_txt', type=str, \
        default='/data/Guoxian_Dai/CUB_200_2011/CUB_200_2011/train_test_split.txt')
parser.add_argument('--label_txt', type=str, \
        default='/data/Guoxian_Dai/CUB_200_2011/CUB_200_2011/image_class_labels.txt')
parser.add_argument('--pretrained_model_path', default='./weights/inception_v3.ckpt', type=str)

parser.add_argument('--pair_type', default='vector', type=str)
parser.add_argument('--mode', default='train', type=str)
parser.add_argument('--with_regularizer', help='whether to use regularizer for parameters', action='store_true')
parser.add_argument('--optimizer', default='rmsprop', type=str)
parser.add_argument('--loss_type', default='contrastive_loss', type=str)
parser.add_argument('--learning_rate_decay_type', default='fixed', type=str)
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--num_workers', default=4, type=int)
parser.add_argument('--num_epochs1', default=20, type=int)
parser.add_argument('--num_epochs2', default=10, type=int)
parser.add_argument('--learning_rate', default=1e-2, type=float)
parser.add_argument('--momentum', default=1e-2, type=float)
parser.add_argument('--learning_rate2', default=1e-4, type=float)
parser.add_argument('--dropout_keep_prob', default=0.5, type=float) 

parser.add_argument('--weight_decay', default=5e-4, type=float) 
parser.add_argument('--restore_ckpt', default=0, type=int)      # 1 for True
parser.add_argument('--evaluation', default=0, type=int)        # 1 for True
parser.add_argument('--weightFile', default='./models/my-model', type=str)
parser.add_argument('--ckpt_dir', default='./models/siamese', type=str)
parser.add_argument('--class_num', default=5, type=int)
parser.add_argument('--targetNum', default=1000, type=int)

parser.add_argument('--margin', default=1.0, type=float)
parser.add_argument('--focal_decay_factor', default=1.0, type=float)
parser.add_argument('--display_step', default=5, type=int, help='step interval for displaying loss')
parser.add_argument('--eval_step', default=5, type=int, help='step interval for evaluate loss')
# image information
parser.add_argument('--width', default=512, type=int)
parser.add_argument('--height', default=512, type=int)

parser.add_argument('--embedding_size', default=128, type=int)
parser.add_argument('--num_epochs_per_decay', default=2, type=int)

args = parser.parse_args()


def main(args):
    # Get the list of filenames and corresponding list of labels for training et validation
    # Get training data

    print(args)

    FocalLoss = model.FocalLoss(root_dir=args.root_dir, image_txt=args.image_txt,
				train_test_split_txt=args.train_test_split_txt,
				label_txt=args.label_txt, batch_size=args.batch_size,
                                optimizer=args.optimizer, loss_type=args.loss_type,
                                learning_rate=args.learning_rate, margin=args.margin,
                                learning_rate_decay_type=args.learning_rate_decay_type,
                                focal_decay_factor=args.focal_decay_factor,
                                with_regularizer=args.with_regularizer,
                                display_step=args.display_step, momentum=args.momentum,
                                eval_step=args.eval_step, embedding_size=args.embedding_size,
                                num_epochs_per_decay=args.num_epochs_per_decay,
                                pair_type=args.pair_type)

    config=tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        if args.mode == 'train':
            FocalLoss.train(sess)
        elif args.mode == 'evaluate':
            FocalLoss.evaluate(sess)


if __name__ == '__main__':
    print("ALL the args information")
    main(args)
