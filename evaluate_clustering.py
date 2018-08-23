"""
This is the module evaluate the clustering performance.
"""

from sklearn.cluster import KMeans

def get_cluster(feature_set, n_clusters=2):
    """"
    Args:
        feature_set: (N, D), N is sample number, D is feature size.
        n_clusters: int, number of clusters to be generated.

    Args:
        kmeams
    """
    kmeans = KMeans(n_clusters=2, random_state=0).fit(feature_set)

    return kmeans


def get_label_statics(label_set):
    unique_labels = np.unique(label_set)
    # get sample number belong to each class
    class_num = unique_labels.shape[0]
    sample_num_per_class = np.zeros(class_num)
    for i in range(class_num):
        index = np.where(label_set == unique_labels[i])
        sample_num_per_class[i] = np.prod(index[0].shape)

    return unique_labels, sample_num_per_class, class_num


def compute_H():

    pass


def evaluate_clutering(feature_set, label_set, n_clusters=None):
    """
    evalute the cluserting performance.
    """
    total_num = label_set[0]
    assert feature_set.shape[0] == label_set[0], \
            "the number of feature doesn't match the number of labels"

    unique_labels, sample_num_per_label, class_num = \
            get_label_statics(label_set)


    if n_clusters is None:
        n_clusters = unique_labels.shape[0]

    kmeans = get_cluster(feature_set, n_clusters)
    cluster_set = kmeans.labels_
    # count number of samples for each cluster
    unique_clusters, sample_num_per_cluster, cluster_num_ = \
            get_label_statics(cluster_set)

    assert n_clusters == cluster_num_, "cluster number doesn't match"

    # count purity
    for i in range(n_clusters):
        # get samples belong to i-the cluster
        index = np.where(cluster_set == unique_clusters[i])
        labels = label_set[index]

        counter = np.zeros(class_num)
        for j in range(class_num):
            index_j = np.where(labels == unique_labels[j])
            counter[j] = index_j[0].shape[0]

        purity += np.amax(counter)


    purity /= float(total_num)
    print("Purity is {:5.3f}".format(purity))

    # compute entropy
















