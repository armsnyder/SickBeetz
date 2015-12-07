from sklearn.cluster import k_means
from scipy.stats.mstats import zscore
from scipy.stats import anderson
import numpy as np


class Clusterer:
    def __init__(self):
        pass

    @staticmethod
    def label(examples):
        labels = g_means(examples)
        print labels
        return labels


def g_means(examples, min_k=1, max_k=10):
    # Still very brute force. Can be optimized.

    labels = []

    # Initialize centers to the overall mean
    centers = 'k-means++'

    k = min_k
    while k <= max_k:
        last_k = k

        # Run the k_means clustering algorithm (the heart of this function):
        centers, labels, _ = k_means(examples, k, centers)

        # Let {xi|class(xi) = j} be the set of datapoints assigned to center cj
        clusters = [{'center': centers[i], 'points':[]} for i in xrange(k)]
        for i in xrange(len(labels)):
            clusters[labels[i]]['points'].append(examples[i])

        # Use a statistical test to detect if each cluster follows a Gaussian distribution with confidence level alpha
        for cluster in clusters:
            if k >= max_k:
                break

            # If the data look Gaussian, keep cj . Otherwise replace cj with two centers.
            is_gaussian, split_centers = check_cluster(cluster['points'])
            if not is_gaussian:
                new_centers = [c for c in centers if not np.array_equal(c, cluster['center'])]
                new_centers.extend(split_centers)
                centers = np.array(new_centers)
                k += 1

        if k == last_k:
            break

    return labels


def check_cluster(cluster):
    n = len(cluster)
    if n < 2:
        return True, []

    # Run k_means on two centers
    children, labels, _ = k_means(cluster, 2)

    # Let v = c1 - c2 be a d-dimensional vector that connects the two centers. This is the direction that k-means
    # believes to be important for clustering.
    v = children[1]-children[0]

    # Then project X onto v: x'i = hxi, vi/||v||2. X0 is a 1-dimensional
    # representation of the data projected onto v.
    x_prime = [np.dot(point, v) for point in cluster]

    # Transform X0 so that it has mean 0 and variance 1.
    x_prime = zscore(x_prime)

    # Let zi = F(x0(i)). If A2*(Z) is in the range of non-critical values at confidence level alpha, then accept H0,
    # keep the original center, and discard {c1, c2}. Otherwise, reject H0 and keep {c1, c2} in place of the original
    # center.
    a2, critical, sig = anderson(x_prime)
    a2 *= (1+4.0/n-25.0/(n**2))

    return a2 < critical[0], children
