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


def inception_net(inputs, num_classes=1000, is_training=False, reuse=False):
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
    logits, end_points = inception.inception_v3(inputs, \
                    num_classes=num_classes, is_training=is_training, \
                    reuse=reuse)

    return logits, end_points


if __name__ == '__main__':
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    inputs = tf.placeholder(tf.float32, [None, 299, 299, 3])
    inputs_ = tf.placeholder(tf.float32, [None, 299, 299, 3])
    is_training = True

    logits, end_points = inception_net(inputs=inputs, \
            is_training=is_training, reuse=False)

    logits_, end_points_ = inception_net(inputs=inputs_, \
            is_training=is_training, reuse=True)


    variables_to_restore = tf.contrib.framework.get_variables_to_restore(\
            exclude=['InceptionV3/Logits/Conv2d_1c_1x1/weights', \
            'InceptionV3/Logits/Conv2d_1c_1x1/biases'])
    init_fn = init_fn = tf.contrib.framework.assign_from_checkpoint_fn('./inception_v3.ckpt', variables_to_restore)
    for var in variables_to_restore:
        print(var.name)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        init_fn(sess)








