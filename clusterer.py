from sklearn.cluster import k_means
from scipy.stats.mstats import zscore


class Clusterer:
    def __init__(self):
        pass

    @staticmethod
    def label(examples, k):
        g_means(examples)  # Work in progress
        centroid, label, inertia = k_means(examples, k)
        return label

def g_means(examples, min_k=1, max_k=10):
    # Work in progress
    centers = 'k-means++'
    for k in xrange(min_k, max_k+1):
        centers, labels, inertia = k_means(examples, k, centers)
        clusters = [[] for _ in xrange(k)]
        for i in xrange(len(labels)):
            clusters[labels[i]].append(examples[i])
        z = [zscore(j) for j in clusters]
        pass