# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 10:04:37 2021

@author: lenovo
"""

import numpy as np

from sklearn.cluster import DBSCAN

from sklearn import metrics

from sklearn.datasets import make_blobs

from sklearn.preprocessing import StandardScaler

# Generatesample data

centers= [[1, 1], [-1, -1], [1, -1]]

X, labels_true =make_blobs(n_samples=750, centers=centers,cluster_std=0.4,

random_state=0)

X=StandardScaler().fit_transform(X)

# Compute DBSCAN

db=DBSCAN(eps=0.3, min_samples=10).fit(X)

core_samples_mask= np.zeros_like(db.labels_,dtype=bool)

core_samples_mask[db.core_sample_indices_]=True

labels= db.labels_

# Number ofclusters in labels, ignoring noise if present.

n_clusters_=len(set(labels)) - (1 if -1 in labels else 0)

n_noise_=list(labels).count(-1)

print('Estimatednumber of clusters: %d'% n_clusters_)

print('Estimated numberof noise points: %d'% n_noise_)

print("Homogeneity:%0.3f"% metrics.homogeneity_score(labels_true,labels))

print("Completeness:%0.3f"% metrics.completeness_score(labels_true,labels))

print("V-measure:%0.3f"% metrics.v_measure_score(labels_true,labels))

print("AdjustedRand Index: %0.3f"

% metrics.adjusted_rand_score(labels_true,labels))

print("AdjustedMutual Information: %0.3f"

% metrics.adjusted_mutual_info_score(labels_true,labels))

print("SilhouetteCoefficient: %0.3f"

% metrics.silhouette_score(X, labels))

# Plot result

import matplotlib.pyplot as plt

#%matplotlib inline

# Black removedand is used for noise instead.

unique_labels=set(labels)

colors= [plt.cm.Spectral(each)


for each in np.linspace(0, 1, len(unique_labels))]
for k, col in zip(unique_labels,colors):
    print (col)
    #col=[1, 1, 1, 1.0]#取消注释则为生成黑白图，否则为每个聚类不同颜色
    #if k ==-1:

# Black used for noise.

    #col = [0, 0, 0, 1]

    class_member_mask = (labels == k)
    xy =X[class_member_mask &core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col), markeredgecolor='k', markersize=14)
    xy =X[class_member_mask &~core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col), markeredgecolor='k', markersize=6)
    plt.title('Estimatednumber of clusters: %d'% n_clusters_)

plt.show()