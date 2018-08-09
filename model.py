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
import time
import numpy as np
import cub_2011


class FocalLoss(object):
    """
    This is for implementation of focal contrastive loss.
    """
    def __init__(self, batch_size=32, dataset_name='cub2011', \
            network_type='siamese_network', pair_type='vector', \
            pretrained_model_path='./weights/inception_v3.ckpt', \
            margin=1., num_epochs_per_decay=2., \
            embedding_size=128, \
            num_epochs=100, \
            learning_rate=0.01, \
            learning_rate_decay_type='exponential', \
            learning_rate_decay_factor=0.94, \
            end_learning_rate=0.0001, \
            optimizer='rmsprop', \
            adam_beta1=0.9, \
            adam_beta2=0.999, \
            opt_epsilon=1., \
            rmsprop_momentum=0.9, \
            rmsprop_decay=0.9, \
            log_dir='./log',
            exclude=['global_step', \
                    'InceptionV3/AuxLogits/Conv2d_2b_1x1/weights', \
                    'InceptionV3/AuxLogits/Conv2d_2b_1x1/biases', \
                    'InceptionV3/Logits/Conv2d_1c_1x1/weights', \
                    'InceptionV3/Logits/Conv2d_1c_1x1/biases'
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
        self.rmsprop_decay = rmsprop_decay
        self.pretrained_model_path = pretrained_model_path
        self.exclude = exclude

        # create iterators
        self.create_iterator()
        self.build_network()





    def build_network(self):
        """
        buld the network.
        """
        self.global_step = tf.Variable(0, name='global_step', trainable=False)


        if self.network_type == 'siamese_network':

            self.features, _ = self.inception_net(self.images, self.embedding_size, \
                    is_training=True, reuse=False)

            self.features_, _ = self.inception_net(self.images_, self.embedding_size, \
                    is_training=True, reuse=True)

            variables_to_restore = tf.contrib.framework.get_variables_to_restore(exclude=self.exclude)

            self.init_fn = tf.contrib.framework.assign_from_checkpoint_fn(self.pretrained_model_path, variables_to_restore)

            pairwise_distances, pairwise_similarity_labels \
                    = self.calculate_distance_and_similarity_label(\
                    self.features, self.features_, self.labels, self.labels_, self.pair_type)

            self.loss = self.contrastive_loss(pairwise_distances, pairwise_similarity_labels)

            self.loss_sum = tf.summary.scalar('loss', self.loss)


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
                    learning_rate_power=FLAGS.ftrl_learning_rate_power,
                    initial_accumulator_value=FLAGS.ftrl_initial_accumulator_value,
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





    def train(self, sess):
        """
        training process.
        """
        learning_rate = self.configure_learning_rate()
        optimizer = self.configure_optimizer(learning_rate)
        train_op = optimizer.minimize(self.loss)
        sess.run(tf.global_variables_initializer())
        self.init_fn(sess)
        counter = 0

        """
        for var in variables_to_restore:
            print(var.name)
        """




        for epoch in range(self.num_epochs):
            print("*"*50 + '\n')
            print("initialize the train_init_op")
            sess.run([self.train_init_op, self.train_init_op_])

            while True:
                try:
                    _, loss_value, loss_sum_ = sess.run([train_op, self.loss, self.loss_sum])
                    counter += 1
                    print("Epoch [{} / {}], step {}, Loss: {}".format(epoch, self.num_epochs, counter, loss_value))

                except tf.errors.OutOfRangeError:
                    break


    def create_iterator(self):
        if self.dataset_name == 'cub2011':
            root_dir = '/raid/Guoxian_Dai/CUB_200_2011/CUB_200_2011/images'
            image_txt = '/raid/Guoxian_Dai/CUB_200_2011/CUB_200_2011/images.txt'
            train_test_split_txt = '/raid/Guoxian_Dai/CUB_200_2011/CUB_200_2011/train_test_split.txt'
            label_txt = '/raid/Guoxian_Dai/CUB_200_2011/CUB_200_2011/image_class_labels.txt'

            train_img_list, train_label_list, test_img_list, test_label_list \
                = cub_2011.generate_list(root_dir, image_txt, train_test_split_txt, label_txt)

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

    def inception_net(self, inputs, num_classes=1000, is_training=False, reuse=False):
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
                    is_training=is_training, reuse=reuse)

        return logits, end_points

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



        if pair_type is None or pair_type == 'matrix':

            # reshape label for convenience
            labels = tf.reshape(labels, [-1, 1])
            labels_ = tf.reshape(labels_, [-1, 1])


            # calcualte pairwise distance
            squared_features = get_squared_features(features)
            squared_features_ = get_squared_features(features_)

            correlation_term = tf.matmul(features, tf.transpose(features_, perm=[1, 0]))

            pairwise_distances = tf.subtract(squared_features + squred_features_, \
                tf.multiply(2., squared_features_))

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

            pairwise_distances = tf.reduce_sum(tf.square(tf.subtract(features, features_)))
            pairwise_similarity_labels = tf.cast(tf.equal(labels, labels_), tf.float32)

            return pairwise_distances, pairwise_similarity_labels


    def contrastive_loss(self, pairwise_distances, pairwise_similarity_labels):
        """
        formulate constrastive loss.
        """

        # positive pair loss
        positive_pair_loss = pairwise_distances * pairwise_similarity_labels
        positive_pair_loss = tf.reduce_mean(positive_pair_loss, name='positive_pair_loss')

        # negative pair loss
        negative_pair_loss = tf.multiply(tf.maximum(tf.subtract(self.margin, \
            pairwise_distances), 0.), tf.subtract(1., pairwise_similarity_labels))
        negative_pair_loss = tf.reduce_mean(negative_pair_loss, name='negative_pair_loss')

        loss = tf.add(positive_pair_loss, negative_pair_loss, name='loss')

        return loss







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

