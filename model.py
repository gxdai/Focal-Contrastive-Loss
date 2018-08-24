import sys
import os
import time
import datetime
import math

import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.nets
from tensorflow.contrib import layers
from tensorflow.contrib.framework.python.ops import arg_scope
from tensorflow.contrib.layers.python.layers import layers as layers_lib
from tensorflow.contrib.layers.python.layers import regularizers
from tensorflow.contrib.layers.python.layers import utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import variable_scope
import numpy as np
from tqdm import tqdm
from scipy.spatial import distance
import cub_2011
import RetrievalEvaluation
import evaluate_recall

import cub_2011
import RetrievalEvaluation

class FocalLoss:
    """
    This is for implementation of focal contrastive loss.
    """
    def __init__(self, batch_size=128, dataset_name='cub2011',
            network_type='siamese_network', pair_type='vector',
            pretrained_model_path='./weights/inception_v3.ckpt',
            margin=1., num_epochs_per_decay=2.,
            root_dir = '/raid/Guoxian_Dai/CUB_200_2011/CUB_200_2011/images',
            image_txt = '/raid/Guoxian_Dai/CUB_200_2011/CUB_200_2011/images.txt',
            train_test_split_txt = '/raid/Guoxian_Dai/CUB_200_2011/CUB_200_2011/train_test_split.txt',
            label_txt = '/raid/Guoxian_Dai/CUB_200_2011/CUB_200_2011/image_class_labels.txt',
            embedding_size=128,
            focal_decay_factor=1.,
            num_epochs=1000,
            learning_rate=0.001,
            learning_rate_decay_type='exponential',
            learning_rate_decay_factor=0.94,
            end_learning_rate=0.0001,
            optimizer='rmsprop',
            adam_beta1=0.9, adam_beta2=0.999,
            opt_epsilon=1.,
            rmsprop_momentum=0.9,
            momentum=0.9,
            rmsprop_decay=0.9,
            log_dir='./log',
            ckpt_dir='checkpoint',
            model_name='model',
            loss_type='contrastive_loss',
            with_regularizer=False,
            display_step=5,
            eval_step=5,
            exclude=['global_step',
                    'InceptionV3/AuxLogits/Conv2d_2b_1x1/weights',
                    'InceptionV3/AuxLogits/Conv2d_2b_1x1/biases',
                    'InceptionV3/Logits/Conv2d_1c_1x1/weights',
                    'InceptionV3/Logits/Conv2d_1c_1x1/biases',
                    'InceptionV3/embedding/Conv2d_1c_1x1/biases',
                    'InceptionV3/embedding/Conv2d_1c_1x1/weights',
                    'InceptionV3/embedding/Conv2d_1d_1x1/biases',
                    'InceptionV3/embedding/Conv2d_1d_1x1/weights'
                ]):
        """
        initialize the class.

        Args:
            batch_size: int
            dataset_name: str


        """
        self.batch_size = batch_size
        self.embedding_size = embedding_size
        self.num_epochs = num_epochs
        self.dataset_name = dataset_name
        self.network_type = network_type
        self.pair_type = pair_type
        self.margin = margin
        self.num_epochs_per_decay = num_epochs_per_decay
        self.learning_rate_decay_type = learning_rate_decay_type
        self.learning_rate_decay_factor = learning_rate_decay_factor
        self.end_learning_rate = end_learning_rate
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.adam_beta1 = adam_beta1
        self.adam_beta2 = adam_beta2
        self.opt_epsilon = opt_epsilon
        self.rmsprop_momentum = rmsprop_momentum
        self.momentum = momentum
        self.rmsprop_decay = rmsprop_decay
        self.pretrained_model_path = pretrained_model_path
        self.exclude = exclude
        self.model_name = model_name
        # whether to add the regulazier for parameters
        self.with_regularizer = with_regularizer
        self.display_step = display_step
        self.eval_step = eval_step

        self.loss_type = loss_type
        self.focal_decay_factor = focal_decay_factor

        if self.loss_type == 'contrastive_loss':
            self.ckpt_dir = os.path.join(ckpt_dir, self.network_type,
                                    self.pair_type, self.loss_type)
        elif self.loss_type  == 'focal_contrastive_loss':
            self.ckpt_dir = os.path.join(ckpt_dir, self.network_type,
                                    self.pair_type, self.loss_type, str(int(self.focal_decay_factor)))
        self.root_dir = root_dir
        self.image_txt = image_txt
        self.train_test_split_txt = train_test_split_txt
        self.label_txt = label_txt

        # create directory
        self.check_and_create_path(self.ckpt_dir)

        self.is_training = tf.placeholder(tf.bool)

        self.k_set = [1, 2, 4, 8]
        # create iterators

        print("Configuration information")
        print("*"*20+'\n')
        print("optimimizer = {:20}".format(self.optimizer))
        print("learning_rate = {:10.5f}".format(self.learning_rate))
        print("learning_rate_decay_type = {:20}".format(self.learning_rate_decay_type))
        print("loss_type = {:20}".format(self.loss_type))
        print("margin = {:20}".format(self.margin))
        print("with_regularizer = {:20}".format(self.with_regularizer))
        print("focal_decay_factor = {:20}".format(self.focal_decay_factor))

        self.create_iterator()
        self.build_network()

        self.saver = tf.train.Saver()

    def build_network(self):
        """
        buld the network.
        """
        self.global_step = tf.Variable(0, name='global_step', trainable=False)

        self.features, _ = self.inception_net(inputs=self.images, embedding_size=self.embedding_size, \
                                              is_training=self.is_training, reuse=False)

        self.features_, _ = self.inception_net(inputs=self.images_, embedding_size=self.embedding_size, \
                                                is_training=self.is_training, reuse=True)



        self.variables_to_train = []
        self.var_list = tf.trainable_variables()
        for var in self.var_list:
            if "embedding" in var.name or "Logits" in var.name:
                self.variables_to_train.append(var)

        variables_to_restore = tf.contrib.framework.get_variables_to_restore(exclude=self.exclude)

        self.init_fn = tf.contrib.framework.assign_from_checkpoint_fn(self.pretrained_model_path, variables_to_restore)

        pairwise_distances, pairwise_similarity_labels \
                            = self.calculate_distance_and_similarity_label(
                                        self.features,
                                        self.features_,
                                        self.labels,
                                        self.labels_,
                                        self.pair_type)


        self.pairwise_distances = tf.identity(pairwise_distances)
        print("\n\t\t************************")
        if self.loss_type == 'contrastive_loss':
            print("\t\t*Build contrastive loss*")
            self.loss, self.positive_pair_loss, self.negative_pair_loss, self.debug_label\
                    = self.contrastive_loss(pairwise_distances, pairwise_similarity_labels)
            self.loss_sum = tf.summary.scalar('loss', self.loss)

        elif self.loss_type == 'focal_contrastive_loss':
            print("\t\t*Build focal contrastive loss*")
            offset = self.margin / 2.
            self.loss, self.positive_pair_loss, self.negative_pair_loss, self.debug_label\
                    = self.focal_contrastive_loss(pairwise_distances, pairwise_similarity_labels,
                                                  focal_decay_factor=self.focal_decay_factor,
                                                  offset=offset)
            self.loss_sum = tf.summary.scalar('loss', self.loss)

        self.regularization_loss = tf.add_n(slim.losses.get_regularization_losses())
        if self.with_regularizer:
            # add regularization loss for parameters
            self.total_loss = self.loss + self.regularization_loss
        else:
            self.total_loss = self.loss

        print("\t\t******************************")


    def configure_learning_rate(self):
        decay_steps = int(self.train_img_num * self.batch_size / \
                self.num_epochs_per_decay)

        if self.learning_rate_decay_type == 'exponential':
            return tf.train.exponential_decay(self.learning_rate,
                    self.global_step,
                    decay_steps,
                    self.learning_rate_decay_factor,
                    staircase=True,
                    name='exponential_decay_learning_rate')
        elif self.learning_rate_decay_type == 'fixed':
            return tf.constant(self.learning_rate, name='fixed_learning_rate')
        elif self.learning_rate_decay_type == 'polynomial':
            return tf.train.polynomial_decay(self.learning_rate,
                    self.global_step,
                    decay_steps,
                    self.end_learning_rate,
                    power=1.,
                    cycle=False,
                    name='polynomial_decay_learning_rate')
        else:
            raise ValueError('learning_rate_decay_type [%s] was not recognized' % \
                    self.learning_rate_decay_type)


    def configure_optimizer(self, learning_rate):
        if self.optimizer == 'adadelta':
            optimizer = tf.train.AdadeltaOptimizer(
                    learning_rate,
                    rho=self.adadelta_rho,
                    epsilon=self.opt_epsilon)
        elif self.optimizer == 'adagrad':
            optimizer = tf.train.AdagradOptimizer(
                    learning_rate,
                    initial_accumulator_value=FLAGS.adagrad_initial_accumulator_value)
        elif self.optimizer == 'adam':
            optimizer = tf.train.AdamOptimizer(
                    learning_rate,
                    beta1=self.adam_beta1,
                    beta2=self.adam_beta2,
                    epsilon=self.opt_epsilon)
        elif self.optimizer == 'ftrl':
            optimizer = tf.train.FtrlOptimizer(
                    learning_rate,
                    learning_rate_power=FLAGS.ftrl_learning_rate_power, initial_accumulator_value=FLAGS.ftrl_initial_accumulator_value,
                    l1_regularization_strength=FLAGS.ftrl_l1,
                    l2_regularization_strength=FLAGS.ftrl_l2)
        elif self.optimizer == 'momentum':
            optimizer = tf.train.MomentumOptimizer(
                    learning_rate,
                    momentum=self.momentum,
                    name='momentum')
        elif self.optimizer == 'rmsprop':
            optimizer = tf.train.RMSPropOptimizer(
                    learning_rate,
                    decay=self.rmsprop_decay,
                    momentum=self.rmsprop_momentum,
                    epsilon=self.opt_epsilon)
        elif self.optimizer == 'sgd':
            optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        else:
            raise ValueError('Optimizer [%s] was not recognized' % \
                    self.optimizer)

        return optimizer


    def check_and_create_path(self, path):
        """
        check the eixstence of path.
        if path doesn't exist, create it.
        """
        if not os.path.isdir(path):
            os.makedirs(path)

    def fc_layer(self,
                inputs,
                embedding_size=128,
                is_training=True,
                dropout_keep_prob=0.8,
                depth_multiplier=1.0,
                prediction_fn=slim.softmax,
                spatial_squeeze=True,
                reuse=None,
                scope='InceptionV3',
                global_pool=False):

        if depth_multiplier <= 0:
            raise ValueError('depth_multiplier is not greater than zero.')
        depth = lambda d: max(int(d * depth_multiplier), min_depth)

        with tf.variable_scope(scope, 'InceptionV3', [inputs], reuse=reuse) as scope:
            with slim.arg_scope([slim.batch_norm, slim.dropout],
                                is_training=is_training):
                                # Final pooling and prediction
                tf.nn.relu
                with tf.variable_scope('embedding'):
                    embedding_vector = slim.conv2d(inputs, 1000, [1, 1], activation_fn=tf.nn.relu,
                                                   normalizer_fn=None, scope='Conv2d_1c_1x1')

                    embedding_vector = slim.conv2d(embedding_vector, embedding_size, [1, 1], activation_fn=None,
                                                   normalizer_fn=None, scope='Conv2d_1d_1x1')
                                                   # weights_initializer=tf.truncated_normal_initializer(stddev=.005))

                if spatial_squeeze:
                    embedding_vector = tf.squeeze(embedding_vector, [1, 2], name='SpatialSqueeze')

        return embedding_vector



    def train(self, sess):
        """
        training process.
        """
        learning_rate = self.configure_learning_rate()
        optimizer = self.configure_optimizer(learning_rate)

        # computer gradients manually
        grads_and_vars = optimizer.compute_gradients(self.loss, self.var_list)

        # Use 10 times learning rate for last two layers
        scale_factor = 10.
        for idx, (grad, var) in enumerate(grads_and_vars):
            if "InceptionV3/embedding" in var.name or "InceptionV3/Logits" in var.name:
                print("***\t"+var.name+"\t******")
                grads_and_vars[idx] = (grad*scale_factor, var)

        # all the training operations

        train_op_ = optimizer.apply_gradients(grads_and_vars)
        train_op_with_last_two_layers = optimizer.minimize(loss=self.total_loss, var_list=self.variables_to_train)
        train_op_with_all_layers = optimizer.minimize(loss=self.total_loss, var_list=self.var_list)
        #####

        sess.run(tf.global_variables_initializer())
        self.init_fn(sess)
        batch_num = int(self.train_img_num / self.batch_size)

        model_file = os.path.join(self.ckpt_dir, self.model_name)
        saver = tf.train.Saver()

        for epoch in range(self.num_epochs):
            sess.run([self.train_init_op, self.train_init_op_])
            batch_idx = 0
            while True:
                try:
                    # for debug
                    """
                    _, loss_value, loss_p, loss_n, loss_sum_, debug_label, label, label_, features \
                            = sess.run([train_op, self.loss, self.positive_pair_loss,
                                        self.negative_pair_loss, self.loss_sum,
                                        self.debug_label, self.labels, self.labels_, self.features],
                                        feed_dict={self.is_training: True})
                    """

                    # = sess.run([train_op_with_last_two_layers if epoch < 3 else train_op_with_last_two_layers,
                    _, total_loss, regu_loss, contrastive_loss, loss_p, \
                            loss_n, loss_sum_, features, features_, distances = sess.run([train_op_,
                                                          self.total_loss, self.regularization_loss, self.loss,
                                                          self.positive_pair_loss, self.negative_pair_loss, self.loss_sum,
                                                          self.features, self.features_, self.pairwise_distances],
                                                          feed_dict={self.is_training: True})
                    batch_idx += 1

                    if batch_idx % self.display_step == 0:
                        print(("{}: Epoch [{:3d}/{:3d}] [{:3d}/{:3d}], total loss: {:6.5f}" +
                               ", regularization loss: {:6.5f}, contrastive loss: {:6.5f}, " +
                               "Loss positive: {:6.5f}, Loss negative: {:6.5f}").format(
                            str(datetime.datetime.now()), epoch, self.num_epochs,
                                batch_idx, batch_num, total_loss, regu_loss,
                                contrastive_loss, loss_p, loss_n))
                        #print(np.sqrt(np.sum(np.square(features), axis=1)))
                        #print(np.sqrt(np.sum(np.square(features_), axis=1)))
                        #print(distances)
                    if math.isnan(total_loss):
                        print("The loss is nan")
                        break

                except tf.errors.OutOfRangeError:
                    break
            if epoch % self.eval_step == 0:
    	        self.evaluate_online(sess, test_mode=2)
            saver.save(sess, model_file, global_step=epoch)

    def get_checkpoint_file(self):
        ckpt = tf.train.get_checkpoint_state(self.ckpt_dir)
        if ckpt and ckpt.model_checkpoint_path:
            return ckpt
        else:
            return None



    def get_feature_and_label_old(init_op, sess, feature_tensor, label_tensor):
            feature_set = []
            label_set = []
            counter = 0
            sess.run(init_op)
            while True:
                try:
                    features, labels = sess.run([self.features, self.labels], feed_dict={self.is_training: False})
                    counter += 1
                    print("Processing the {:3d}-th batch".format(counter))
                except tf.errors.OutOfRangeError:
                    break

                feature_set.append(features)
                label_set.append(labels)

            feature_set = np.concatenate(feature_set, axis=0)
            label_set = np.concatenate(label_set, axis=0)

            return feature_set, label_set


    def get_feature_and_label(self, init_op, sess, feature_tensor, label_tensor, batch_num):
        feature_set = []
        label_set = []
        sess.run(init_op)
        while True:
            try:
                features, labels = sess.run([self.features, self.labels], feed_dict={self.is_training: False})
            except tf.errors.OutOfRangeError:
                break
            feature_set.append(features)
            label_set.append(labels)

        feature_set = np.concatenate(feature_set, axis=0)
        label_set = np.concatenate(label_set, axis=0)

        return feature_set, label_set

    def evaluate_online(self, sess, test_mode=1):

        if test_mode == 1:
            train_feature_set, train_label_set = \
                get_feature_and_label(self.train_init_op, sess, self.features, self.labels)


            test_feature_set, test_label_set = \
                get_feature_and_label(self.test_init_op, sess, self.features, self.labels)


            distM = distance.cdist(test_feature_set, train_feature_set)
            nn_av, ft_av, st_av, dcg_av, e_av, map_, p_points, pre, rec, rankArray = RetrievalEvaluation.RetrievalEvaluation(distM, train_label_set, test_label_set, testMode=1)

        elif test_mode == 2:

            batch_num = int(self.test_img_num / self.batch_size) + 2
            test_feature_set, test_label_set = \
                self.get_feature_and_label(self.test_init_op, sess, self.features, self.labels, batch_num)

            distM = distance.cdist(test_feature_set, test_feature_set)
            nn_av, ft_av, st_av, dcg_av, e_av, map_, p_points, pre, rec, rankArray = RetrievalEvaluation.RetrievalEvaluation(distM, test_label_set, test_label_set, testMode=2)

            recall_k = evaluate_recall.evaluate_recall(test_feature_set, test_label_set, self.k_set)
            for i, k in enumerate(self.k_set):
                print("Recall@{:1d}: {:6.5f}".format(self.k_set[i], recall_k[i]))



        print(('The NN is {:5.5f}\nThe FT is {:5.5f}\n' +
              'The ST is {:5.5f}\nThe DCG is {:5.5f}\n' +
              'The E is {:5.5f}\nThe MAP {:5.5f}\n').format(
                  nn_av, ft_av, st_av, dcg_av, e_av, map_))





    def evaluate(self, sess, test_mode=2):
        ckpt = self.get_checkpoint_file()
        sess.run(tf.global_variables_initializer())
        if ckpt is None:
            raise IOError("No check point file found")
        else:
            self.saver.restore(sess, ckpt.model_checkpoint_path)

        def get_feature_and_label(init_op, sess, feature_tensor, label_tensor):
            feature_set = []
            label_set = []
            counter = 0
            sess.run(init_op)
            while True:
                try:
                    features, labels = sess.run([self.features, self.labels], feed_dict={self.is_training: False})
                    counter += 1
                    print("Processing the {:3d}-th batch".format(counter))
                except tf.errors.OutOfRangeError:
                    break

                feature_set.append(features)
                label_set.append(labels)

            feature_set = np.concatenate(feature_set, axis=0)
            label_set = np.concatenate(label_set, axis=0)

            return feature_set, label_set


        if test_mode == 1:
            train_feature_set, train_label_set = \
                get_feature_and_label(self.train_init_op, sess, self.features, self.labels)


            test_feature_set, test_label_set = \
                get_feature_and_label(self.test_init_op, sess, self.features, self.labels)


            distM = distance.cdist(test_feature_set, train_feature_set)
            nn_av, ft_av, st_av, dcg_av, e_av, map_, p_points, pre, rec, rankArray = RetrievalEvaluation.RetrievalEvaluation(distM, train_label_set, test_label_set, testMode=1)

        elif test_mode == 2:

            test_feature_set, test_label_set = \
                get_feature_and_label(self.test_init_op, sess, self.features, self.labels)
            distM = distance.cdist(test_feature_set, test_feature_set)
            nn_av, ft_av, st_av, dcg_av, e_av, map_, p_points, pre, rec, rankArray = RetrievalEvaluation.RetrievalEvaluation(distM, test_label_set, test_label_set, testMode=2)

            recall_k = evaluate_recall.evaluate_recall(test_feature_set, test_label_set, self.k_set)

            for i, k in enumerate(self.k_set):
                print("Recall@{:1d}: {:6.5f}".format(self.k_set[i], recall_k[i]))



        print(('The NN is {:5.5f}\nThe FT is {:5.5f}\n' +
              'The ST is {:5.5f}\nThe DCG is {:5.5f}\n' +
              'The E is {:5.5f}\nThe MAP {:5.5f}\n').format(
                  nn_av, ft_av, st_av, dcg_av, e_av, map_))



    def evaluate_online_backup(self, sess, test_mode=1):

        def get_feature_and_label(init_op, sess, feature_tensor, label_tensor):
            feature_set = []
            label_set = []
            counter = 0
            sess.run(init_op)
            while True:
                try:
                    features, labels = sess.run([self.features, self.labels], feed_dict={self.is_training: False})
                    counter += 1
                    print("Processing the {:3d}-th batch".format(counter))
                except tf.errors.OutOfRangeError:
                    break

                feature_set.append(features)
                label_set.append(labels)

            feature_set = np.concatenate(feature_set, axis=0)
            label_set = np.concatenate(label_set, axis=0)

            return feature_set, label_set


        if test_mode == 1:
            train_feature_set, train_label_set = \
                get_feature_and_label(self.train_init_op, sess, self.features, self.labels)


            test_feature_set, test_label_set = \
                get_feature_and_label(self.test_init_op, sess, self.features, self.labels)


            distM = distance.cdist(test_feature_set, train_feature_set)
            nn_av, ft_av, st_av, dcg_av, e_av, map_, p_points, pre, rec, rankArray = RetrievalEvaluation.RetrievalEvaluation(distM, train_label_set, test_label_set, testMode=1)

        elif test_mode == 2:

            test_feature_set, test_label_set = \
                get_feature_and_label(self.test_init_op, sess, self.features, self.labels)
            distM = distance.cdist(test_feature_set, test_feature_set)
            nn_av, ft_av, st_av, dcg_av, e_av, map_, p_points, pre, rec, rankArray = RetrievalEvaluation.RetrievalEvaluation(distM, test_label_set, test_label_set, testMode=2)




    def evaluate_backup(self, sess):
        """
        training process.
        """

        """
        for var in variables_to_restore:
            print(var.name)
        """

        print("Process training data")
        train_feature_set = []
        train_label_set = []
        batch_num = int(self.test_img_num / self.batch_size)

        ckpt = self.get_checkpoint_file()
        sess.run(tf.global_variables_initializer())
        sess.run(self.train_init_op)
        if ckpt is None:
            raise IOError("No check point file found")
        else:
            self.saver.restore(sess, ckpt.model_checkpoint_path)

        # counting batches
        batch_idx = 0
        batch_num = int(self.train_img_num / self.batch_size)
        while True:

            try:


                features, labels = sess.run([self.features, self.labels])

            except tf.errors.OutOfRangeError:
                break

            batch_idx += 1
            train_feature_set.append(features)
            train_label_set.append(labels)
            print("Processing batch [{:3d}/{:3d}]".format(batch_idx, batch_num))

        train_feature_set = np.concatenate(train_feature_set, axis=0)

        # label should be int type
        train_label_set = np.concatenate(train_label_set, axis=0).astype(np.int)

        print("Finish processing training data")

        print("Process test data")
        test_feature_set = []
        test_label_set = []
        batch_num = int(self.test_img_num / self.batch_size)

        sess.run(self.test_init_op)
        # counting batches
        batch_idx = 0
        batch_num = int(self.test_img_num / self.batch_size)
        while True:
            try:

                features, labels = sess.run([self.features, self.labels])
                batch_idx += 1
            except tf.errors.OutOfRangeError:
                break

            test_feature_set.append(features)
            test_label_set.append(labels)
            print("Processing batch [{:3d}/{:3d}]".format(batch_idx, batch_num))


        test_feature_set = np.concatenate(test_feature_set, axis=0)

        # label should be int type.
        test_label_set = np.concatenate(test_label_set, axis=0).astype(np.int)
        print("Finish processing testing data")

        dist_matrix = distance.cdist(test_feature_set, train_feature_set)
        nn_av, ft_av, st_av, dcg_av, e_av, map_, p_points, pre, rec, rankArray = \
                RetrievalEvaluation.RetrievalEvaluation(dist_matrix, train_label_set, test_label_set, testMode=1)

        print ('The NN is %5f' % (nn_av))
        print ('The FT is %5f' % (ft_av))
        print ('The ST is %5f' % (st_av))
        print ('The DCG is %5f' % (dcg_av))
        print ('The E is %5f' % (e_av))
        print ('The MAP is %5f' % (map_))


    def create_iterator(self):
        if self.dataset_name == 'cub2011':
            """
            root_dir = '/raid/Guoxian_Dai/CUB_200_2011/CUB_200_2011/images'
            image_txt = '/raid/Guoxian_Dai/CUB_200_2011/CUB_200_2011/images.txt'
            train_test_split_txt = '/raid/Guoxian_Dai/CUB_200_2011/CUB_200_2011/train_test_split.txt'
            label_txt = '/raid/Guoxian_Dai/CUB_200_2011/CUB_200_2011/image_class_labels.txt'
            """
            train_img_list, train_label_list, test_img_list, test_label_list \
                = cub_2011.generate_half_split_list(self.root_dir, self.image_txt,
					self.train_test_split_txt, self.label_txt)
            # prepare the total image numbers
            self.train_img_num = len(train_img_list)
            self.test_img_num = len(test_img_list)

            # initalize the dataset
            self.images, self.labels, self.train_init_op, self.test_init_op = \
                cub_2011.create_dataset(train_img_list, train_label_list, \
                    test_img_list, test_label_list, self.batch_size)

            # initalize the dataset
            self.images_, self.labels_, self.train_init_op_, self.test_init_op_ = \
                cub_2011.create_dataset(train_img_list, train_label_list, \
                    test_img_list, test_label_list, self.batch_size)

    def inception_net(self, inputs, num_classes=1000,
                      embedding_size=128, is_training=False,
                      reuse=False):
        """
        Args:
            model_name: "inception"
            inputs: NWHC
            is_training: True or False
            reuse: True or False

        Returns:
            logits: the network output
            end_points: a dictionary for saving all the outputs for all layers.
        """
        # Default model is inceptionNet

        inception = tensorflow.contrib.slim.nets.inception
        with slim.arg_scope(slim.nets.inception.inception_v3_arg_scope()):
            logits, end_points = inception.inception_v3(inputs, num_classes=num_classes,
                                                        is_training=is_training, reuse=reuse,
                                                        spatial_squeeze=False)

            embedding_vector = self.fc_layer(logits, embedding_size=embedding_size,
                                            is_training=is_training, reuse=reuse,
                                            spatial_squeeze=True)

        return embedding_vector, end_points

    def calculate_distance_and_similarity_label(self, features, features_, labels, labels_, pair_type):

        """
        The calculate is based on following equations
        X: (N, M)
        Y: (P, M)

        Each row represents one sample.

        the pairwise distance between X and Y is formulated as


        TO BE CONTINUED.


        Args:
            features: (N, M)
            features_: (N, M)
            labels: (N,)
            labels_: (N,)
            pair_type: str
                "vector":   generating N pairs
                "matrix":   generating N^2 pairs

        Returns:
            pairwise_distances: (N,) for "vector", (N, N) for "matrix"
            pairwise_similarity_labels:   (N,) for "vector", (N, N) for "matrix"

        """

        def get_squared_features(features):
            """
            elementwised operation.
            """
            return tf.expand(tf.reduce_sum(tf.square(features), axis=1), axis=1)


        print("\n\t\t***********************")
        if pair_type is None or pair_type == 'matrix':
            print("\t\tthe pair_type is matrix")

            print("\t\t***********************\n")
            # reshape label for convenience
            labels = tf.reshape(labels, [-1, 1])
            labels_ = tf.reshape(labels_, [-1, 1])


            # calcualte pairwise distance
            squared_features = get_squared_features(features)
            squared_features_ = get_squared_features(features_)

            correlation_term = tf.matmul(features, tf.transpose(features_, perm=[1, 0]))

            pairwise_distances = tf.sqrt(tf.subtract(squared_features + squred_features_, \
                tf.multiply(2., squared_features_)))

            # calcualte pairwise similarity labels
            num_labels = tf.shape(labels)[0]
            num_labels_ = tf.shape(labels_)[0]
            tiled_labels = tf.tile(labels, [1, num_labels_])
            tiled_labels_ = tf.tile(labels_, [num_labels, 1])


            pairwise_similarity_labels = tf.cast(tf.equal(tf.reshape(tiled_labels, [-1]), \
                tf.reshape(tiled_labels_, [-1])), tf.float32)
            pairwise_similarity_labels = tf.reshape(pairwise_similarity_labels, [num_labels, num_labels_])


            return pairwise_distances, pairwise_similarity_labels

        elif pair_type == 'vector':

            print("\t\tthe pair_type is vector")

            print("\t\t***********************\n")
            pairwise_distances = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(features, features_)), axis=1))
            pairwise_similarity_labels = tf.cast(tf.equal(labels, labels_), tf.float32)

            return pairwise_distances, pairwise_similarity_labels


    def contrastive_loss(self, pairwise_distances, pairwise_similarity_labels):
        """
        formulate constrastive loss.

            L_{ij} = Y_{ij} * d_{ij}^2 + (1 - Y_{ij}) * (max(h - d_{ij}, 0.))^2

            Loss

            where d_{ij} is Euclidean distance for pair (i, j)
            Y_{ij} is similarity label. 1 for similar pair, 0 for non-similar pair.
        """

        # positive pair loss
        positive_pair_loss = tf.square(pairwise_distances) * pairwise_similarity_labels
        positive_pair_loss = tf.reduce_mean(positive_pair_loss, name='positive_pair_loss')

        # negative pair loss
        negative_pair_loss = tf.multiply(tf.square(tf.maximum(tf.subtract(self.margin, \
            pairwise_distances), 0.)), tf.subtract(1., pairwise_similarity_labels))
        negative_pair_loss = tf.reduce_mean(negative_pair_loss, name='negative_pair_loss')

        loss = tf.add(positive_pair_loss, negative_pair_loss, name='loss')

        return loss, positive_pair_loss, negative_pair_loss, pairwise_similarity_labels

    def focal_contrastive_loss(self, pairwise_distances,
                                pairwise_similarity_labels,
                                focal_decay_factor=1.,
                                offset=0.5):
        """
        formulate focal constrastive loss.

            L_{ij} = Prob_{ij} * Y_{ij} * d_{ij}^2 + Prob_{ij} * (1 - Y_{ij}) * (max(h - d_{ij}, 0.))^2


            where   d_{ij} is Euclidean distance for pair (i, j)
                    Y_{ij} is similarity label. 1 for similar pair, 0 for non-similar pair.

                    P_{ij} = sigmoid(d)
        """
        def linear_transformation(distances, offset, decay_factor):
            """
            apply a linear transformation to the distances.

            return (distances - offset) / decay_factor

            """
            return tf.divide(tf.subtract(distances,  offset), decay_factor)

        # Apply a linear transformation for positive pairwise distance
        positive_pairwise_distances_transformed = \
                tf.divide(tf.subtract(pairwise_distances, offset), focal_decay_factor)

        # convert distance into probability
        positive_pairwise_prob = 2. * tf.sigmoid(positive_pairwise_distances_transformed)
        # positive pair loss
        positive_pair_loss = positive_pairwise_prob * tf.square(pairwise_distances) \
                * pairwise_similarity_labels

        positive_pair_loss = tf.reduce_mean(positive_pair_loss, name='positive_pair_loss')

        # negative pairwise distance
        negative_pairwise_distances = tf.maximum(tf.subtract(self.margin, \
            pairwise_distances), 0.)

        # Apply a linear transformation for negative pairwise distance
        negative_pairwise_distance_transformed = \
                tf.divide(tf.subtract(
                    negative_pairwise_distances, offset), focal_decay_factor)

        # convert distance into probability
        negative_pairwise_prob = 2. * tf.sigmoid(negative_pairwise_distance_transformed)

        # negative pair loss
        negative_pair_loss = negative_pairwise_prob * tf.square(negative_pairwise_distances) *\
                tf.subtract(1., pairwise_similarity_labels)

        negative_pair_loss = tf.reduce_mean(negative_pair_loss, name='negative_pair_loss')
        loss = tf.add(positive_pair_loss, negative_pair_loss, name='loss')

        return loss, positive_pair_loss, negative_pair_loss, pairwise_similarity_labels


if __name__ == '__main__':
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    inputs = tf.placeholder(tf.float32, [None, 299, 299, 3])
    inputs_ = tf.placeholder(tf.float32, [None, 299, 299, 3])
    is_training = True
    num_classes = 1001
    logits, end_points = inception_net(inputs=inputs, num_classes=num_classes, \
            is_training=is_training, reuse=False)

    logits_, end_points_ = inception_net(inputs=inputs_, num_classes=num_classes, \
            is_training=is_training, reuse=True)

    variables_to_restore = tf.contrib.framework.get_variables_to_restore()

    saver = tf.train.Saver()

    init_fn = tf.contrib.framework.assign_from_checkpoint_fn('./inception_v3.ckpt', variables_to_restore)
    for var in variables_to_restore:
        print(var.name)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        init_fn(sess)
        for _ in range(100):
            out = sess.run(logits, feed_dict={inputs: np.random.random((5, 299, 299, 3))})
            print(out.shape)

