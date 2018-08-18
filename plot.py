"""
Plot focal contrastive loss.
"""
import numpy as np
import matplotlib.pyplot as plt
fig = plt.figure()
ax = plt.subplot(121)
# marginal distance for negative pairs.
margin =  1.
# step for ploting curves.
step = 0.02
# sigma control the weight decay rate.
sigma_1 = 0.1 
sigma_2 = 0.15
end_point = 5.
dist = np.arange(0., end_point, step) 
offset = margin/2.
def weight_func(input_x, offset, sigma):
    """
    Args:
        input_x: x axis data
        margin: marginal distance for negative pairs.
        sigma:  decay rate for weight func

    Returns:
        the weighted distace
    """

    return  2. / (1. + np.exp(-1. * (input_x - offset) / sigma))

weight = weight_func(dist, offset, sigma_1)
weight_2 = weight_func(dist, offset, sigma_2)

origin_dist = np.square(dist)
weighted_dist =  weight * np.square(dist)
weighted_dist_2 =  weight_2 * np.square(dist)


# plot results
ax.plot(dist, origin_dist, label=r'$\sigma=0$')
ax.plot(dist, weighted_dist, label=r'$\sigma=0.1$')
ax.plot(dist, weighted_dist_2, label=r'$\sigma=0.2$')
ax.legend(loc='upper left')

ax.set_xlabel("pairwise distances")
ax.set_ylabel("Loss for positive pair examples")

# for negative loss
ax = plt.subplot(122)
dist_negative = np.maximum(0, margin - dist)

origin_dist_negative = np.square(dist_negative)
weighted_dist_negative = weight_func(dist_negative, offset, sigma_1) * origin_dist_negative
weighted_dist_negative_2 = weight_func(dist_negative, offset, sigma_2) * origin_dist_negative


# plot results
ax.plot(dist, origin_dist_negative, label=r'$\sigma=0$')
ax.plot(dist, weighted_dist_negative, label=r'$\sigma=0.1$')
ax.plot(dist, weighted_dist_negative_2, label=r'$\sigma=0.2$')
ax.legend(loc='upper right')

ax.set_xlabel("pairwise distances")
ax.set_ylabel("Loss for negative pair examples")

plt.show()
