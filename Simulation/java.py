#!/usr/local/opt/python/libexec/bin/python

import csv
import os
import numpy as np
from random import seed
from random import random
from sklearn import metrics
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans

device_list = []
print_list = []
# path to data
filename = '/Users/laric/Documents/eclipse-workspace/FastSynchronization/clus.txt'
counter = 0


def main():
    with open(filename) as f:
        lines = f.read().splitlines()
    for i in range(len(lines)):
        # dev_ = "D" + str(i+1)
        dev_ = str(i + 1)
        device_list.append(dev_)
        new_list = list(zip(device_list, np.zeros(len(device_list))))
        # convert list of devices to numpy array
        np_dev_list = np.asarray(new_list)
    dbscan(lines, np_dev_list)
    # k_Means(lines, np_dev_list)


def dbscan(X, np_dev_list):
    min_samples = 3  # dbscan min samples
    eps = 0.075  # dbscan epsilon value

    Y = list(zip(X, np.zeros(len(X))))
    # print(Y)
    x = np.asarray(Y)
    x = x.astype(np.float64)

    db_default = DBSCAN(min_samples=min_samples, eps=eps).fit(x)
    labels = db_default.labels_
    # clusters = db_default.fit_predict(X)
    labels_unique = np.unique(labels)
    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)
    print("===================DBSCAN==================")

    # print(labels)
    print('number of clusters,%d' % n_clusters_)
    print('noise points,%d' % n_noise_)

    # for k in range(-1, len(labels_unique)-1, 1):
    #     my_members = labels == k
    #     #print("cluster{0}: {1} ==> {2}".format(k, x[my_members, 0],
    #                                             #len(x[my_members, 0])))
    #     print("cluster{0}: {1}={2}".format(k, np_dev_list[my_members,0],
    #                                             len(np_dev_list[my_members, 0])))
    #     print_list.append(np_dev_list[my_members, 0])

    for k in range(len(labels_unique)):
        my_members = labels == k
        # print("cluster{0}: {1} ==> {2}".format(k, x[my_members, 0],
        # len(x[my_members, 0])))
        # print("cluster{0}: {1}={2}".format(k, np_dev_list[my_members,0],
        # len(np_dev_list[my_members, 0])))
        print_list.append(np_dev_list[my_members, 0])

    print_list.sort(key=len, reverse=True)
    for i in range(len(print_list)):
        # print("{0},{1}".format((*print_list[i], sep=", "), len(print_list[i])))
        print(*print_list[i], sep=",")

    for k in range(len(labels_unique)):
        my_members = labels == k
        print("cluster{0}: {1} ==> {2}".format(k, x[my_members, 0],
                                               len(x[my_members, 0])))


def k_Means(y, np_dev_list):
    # Specify the number of clusters (3) and fit the data X

    y1 = list(zip(y, np.zeros(len(y))))
    y2 = np.asarray(y1)
    x, out = detectOutliers(y2, np_dev_list)

    # x = X
    # np_dev_list = out
    k_means = KMeans(n_clusters=2, random_state=2, init='k-means++').fit(x)

    # Get the cluster centroids
    labels = k_means.labels_
    # cluster_centers = k_means.cluster_centers_

    # Get the cluster labels
    labels_unique = np.unique(labels)
    n_clusters_ = len(labels_unique)
    # cluster_list.append(labels)
    # print("===================K-Means==================")
    n_noise_ = len(np_dev_list) - len(out)
    print('number of clusters,%d' % n_clusters_)
    print('noise points,%d' % n_noise_)

    for k in range(len(labels_unique)):
        my_members = labels == k
        # print("cluster{0}: {1} ==> {2}".format(k, x[my_members, 0],
        # len(x[my_members, 0])))
        # print("cluster{0}: {1}={2}".format(k, np_dev_list[my_members,0],
        # len(np_dev_list[my_members, 0])))
        print_list.append(out[my_members, 0])
        print_list.sort(key=len, reverse=True)

    # print_list.sort(key=myFunc, reverse=True)
    for i in range(len(print_list)):
        # print("{0},{1}".format((*print_list[i], sep=", "), len(print_list[i])))
        print(*print_list[i], sep=",")

    clusters = []

    for k in range(len(labels_unique)):
        my_members = labels == k
        print("cluster{0}: {1} ==> {2}".format(k, x[my_members, 0],
                                               len(x[my_members, 0])))


def detectOutliers(t, dev_list):
    t3_1, t3_2 = zip(*t)
    t3_list = list(t3_1)

    out_1, out_2 = zip(*dev_list)
    out_list = list(out_1)

    t3_list = np.asarray(t3_list, dtype='float64')
    t1 = []
    out1 = []
    i: int = 0
    q1 = np.percentile(t3_list, 25, interpolation='midpoint')
    q3 = np.percentile(t3_list, 75, interpolation='midpoint')
    iqr = q3 - q1
    # upper, lower = q3 + 1.5 * iqr, q1 - 1.5 * iqr
    upper, lower = q3 + iqr, q1 - iqr

    for j in range(len(t3_list)):
        if upper > t3_list[j] > lower:
            t1.append(t3_list[j])
            out1.append(out_list[j])
        else:
            i = i + 1

    # set_difference = set(t3_list) - set(t1)
    # list_difference = list(set_difference)
    # print(list_difference)
    return np.asarray(list(zip(t1, np.zeros(len(t1))))), np.asarray(list(zip(out1, np.zeros(len(out1)))))


main()
