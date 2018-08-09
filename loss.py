"""
This module is for contructing loss function.
"""

import tensorflow as tf

def calculate_distance_and_similariy_label(features, features_, labels, labels_, pair_type):
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


    # reshape label for convenience

    if pair_type is None or pair_type == 'matrix':


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


def contrastive_loss(pairwise_distances, pairwise_similarity_labels, margin):
    """
    formulate constrastive loss.
    """

    # positive pair loss
    positive_pair_loss = pairwise_distances * pairwise_similarity_labels
    positive_pair_loss = tf.reduce_mean(positive_pair_loss, name='positive_pair_loss')

    # negative pair loss
    negative_pair_loss = tf.multiply(tf.maximium(tf.subtract(margin, \
            pairwise_distances), 0.), tf.subtract(1., pairwise_similarity_labels))
    negative_pair_loss = tf.reduce_mean(negative_pair_loss, name='negative_pair_loss')

    loss = tf.add(positive_pair_loss, negative_pair_loss, name='loss')

    return loss









