import csv
import os
import numpy as np
from sklearn.cluster import KMeans
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn import metrics
from sklearn.cluster import DBSCAN
from matplotlib import pyplot as plt

MIN_SAMPLES = 3  # dbscan min samples
EPS = 0.0275  # dbscan epsilon value
DIVISOR = 234  # divisor for scaling
OVERLAP = 1000
STEP = 150
STOP = 10500
# array to hold maximum range within clusers
max_inter_cluster = []
intra_cluster_distance = []
# array to hold deviation from pivot
global_array = []
# array to hold pivot data
pivot_array = []
# array to hold sum of execution time values
b = []
# array to hold device names
device_list = []
# array to hold cluster values for each device
cluster_list = []
cluster_list2 = []
clus_list2 = []
cluster_list3 = []
clus_list3 = []
cluster_list4 = []
clus_list4 = []
# array to hold updated cluster list
cluster_list_new = []
# path to directory for data
dir_path = "/Users/laric/Documents/NewResults/"
# path to pivot data
dir_path_pivot = "/Users/laric/Documents/NewResults/v.csv"
# Holds values for num of devices per cluster
second = []
# 2-clusters counter
two_count: int = 0
big_twos_ = []
# 3-clusters counter
three_count: int = 0
big_threes_ = []
# 4-clusters counter
four_count: int = 0
big_fours_ = []
# 5-clusters counter
five_count: int = 0

num_clusters = []
max_dist = 0

range2 = []
average2 = []
range3 = []
average3 = []
range4 = []
average4 = []

font = {'family': 'serif',
        'size': 18}

combined_list_to_plot = []

# create devices
for c in range(37):
    name = "Device" + str(c + 1)
    device_list.append(name)

    new_list = list(zip(device_list, np.zeros(len(device_list))))
    # convert list of devices to numpy array
    np_dev_list = np.asarray(new_list)


def main():
    # get the deviations for each device
    tempList = get_devices_overlap()

    for i in range(70):
        # if i != 0 and i != 3 and i != 4:
        if 1 == 1:
            # if i != 0 and i != 1 and i != 7:
            print('Clustering point {0}'.format(i))
            y = list(zip(tempList[i], np.zeros(len(tempList[i]))))
            # print(Y)
            x = np.asarray(y)
            dbscan(x)
            # k_Means(x)
            # affinity_propagation(x)
            # mean_shift(x)

    for j in range(len(clus_list2)):
        if j != 0:
            cluster_list2.append(clus_list2[j])

    for j in range(len(clus_list3)):
        if j != 0 and j != 5 and j != 8:
            cluster_list3.append(clus_list3[j])

    for j in range(len(clus_list4)):
        if j != 16 and j != 17:
            cluster_list4.append(clus_list4[j])

    distances()
    go_to_all_to_all()
    # go_to_process()
    # plotHisto()
    # process_cluster_all_to_all(cluster_list, 3)
    num_clusters.sort()
    d = {z: num_clusters.count(z) for z in num_clusters}
    print(d)
    checker()
    # print('Two clusters: {0} \nThree clusters: {1} \nFour clusters: {2}'.
    #       format(len(cluster_list2), len(cluster_list3), len(cluster_list4)))


def distances():
    range2_mean_ = np.mean(range2)
    range3_mean_ = np.mean(range3)
    range4_mean_ = np.mean(range4)

    average2_mean_ = np.mean(average2)
    average3_mean_ = np.mean(average3)
    average4_mean_ = np.mean(average4)

    range2_std_ = np.std(range2)
    range3_std_ = np.std(range3)
    range4_std_ = np.std(range4)

    average2_std_ = np.std(average2)
    average3_std_ = np.std(average3)
    average4_std_ = np.std(average4)

    # print(range2_mean_)
    # print(range3_mean_)
    # print(range4_mean_)

    # Create lists for the plot
    labels = ['2-Clusters', '3-Clusters', '4-Clusters']
    x_pos = np.arange(len(labels))
    CTEs_range = [range2_mean_, range3_mean_, range4_mean_]
    error_range = [range2_std_, range3_std_, range4_std_]
    CTEs_average = [average2_mean_, average3_mean_, average4_mean_]
    error_average = [average2_std_, average3_std_, average4_std_]
    width = 0.2

    # Build the plot
    fig, ax = plt.subplots()
    p1 = ax.bar(x_pos, CTEs_range, width=width, yerr=error_range,
                align='center', capsize=10,
                color='lightgray',  hatch = 'o', edgecolor = 'black')

    p2 = ax.bar(x_pos + width, CTEs_average, width=width, yerr=error_average,
                align='center', capsize=10, color='lightblue',
                hatch = 'x', edgecolor = 'black')
    ax.set_xlabel('Clusters per clustering point', fontsize=18)
    ax.set_ylabel('Distance', fontsize=18)
    ax.set_xticks(x_pos + width / 2)
    ax.set_xticklabels(labels, fontsize=18)
    ax.tick_params(axis='y', labelsize=16)
    ax.legend((p1[0], p2[0]), ('Intra-cluster distance', 'Inter-cluster distance'),
              fontsize=18)
    # ax.yaxis.grid(True)

    # Save the figure and show
    plt.tight_layout()
    plt.rc('font', **font)
    #plt.savefig('Results/inter_cluster_new_june3.pdf')
    plt.show()

    two_t = transpose(big_twos_)
    three_t = transpose(big_threes_)
    four_t = transpose(big_fours_)

    two_tt = [np.mean(z) for z in two_t]
    three_tt = [np.mean(z) for z in three_t]
    four_tt = [np.mean(z) for z in four_t]

    two_tt_ = [np.std(z) for z in two_t]
    three_tt_ = [np.std(z) for z in three_t]
    four_tt_ = [np.std(z) for z in four_t]

    y2 = [z for z in two_tt if z != 0]
    y3 = [z for z in three_tt if z != 0]
    y4 = [z for z in four_tt if z != 0]

    y2_ = [z for z in two_tt_ if z != 0]
    y3_ = [z for z in three_tt_ if z != 0]
    y4_ = [z for z in four_tt_ if z != 0]

    # print(two_ttt)
    # print(three_ttt)
    # print(four_ttt)

    # plot for number of devices per cluster
    width_ = 1
    group_gap_ = 1

    labels_ = ['C1', 'C2', 'OUT', 'C1', 'C2', 'C3', 'OUT',
                'C1', 'C2', 'C3', 'C4', 'OUT']

    x2 = np.arange(len(y2))
    x3 = np.arange(len(y3)) + group_gap_ + len(y2)
    x4 = np.arange(len(y4)) + group_gap_ + len(y3) + group_gap_ + len(y2)
    ind = np.concatenate((x2, x3, x4))
    print(ind)

    p1 = ax.bar(x_pos, CTEs_range, width=width, yerr=error_range,
                align='center', capsize=10)

    fig, ax = plt.subplots()
    rects1 = ax.bar(x2, y2, width_, edgecolor="black", label="2-Clusters",
                    yerr=y2_, align='center', capsize=10, color='lightgray',  hatch = 'o')
    rects2 = ax.bar(x3, y3, width_, edgecolor="black", label="3-Clusters",
                    yerr=y3_, align='center', capsize=10,
                    color='lightblue',  hatch = 'o')
    rects3 = ax.bar(x4, y4, width_, edgecolor="black", label="4-Clusters",
                    yerr=y4_, align='center', capsize=10,
                    color='white',  hatch = '/')
    ax.set_ylabel('Number of Devices', fontsize=18, fontweight='bold')
    ax.set_xticks(ind)
    #ax.set_xticklabels('C1', 'C2', 'OUT', 'C1', 'C2', 'C3', 'OUT',
                        #'C1', 'C2', 'C3', 'C4', 'OUT')
    ax.set_xticklabels(labels_, fontsize=13.5, fontweight='bold')
    ax.tick_params(axis='y', labelsize=16)
    ax.legend(loc=1, fontsize=18)

    # Save the figure and show
    plt.tight_layout()
    plt.rc('font', **font)
    #plt.savefig('Results/num_devices_new_june3.pdf')
    plt.show()


def go_to_all_to_all():
    print('Processing 2 clusters')
    process_cluster_all_to_all(cluster_list2, 2)
    print('Processing 3 clusters')
    process_cluster_all_to_all(cluster_list3, 3)
    print('Processing 4 clusters')
    process_cluster_all_to_all(cluster_list4, 4)


def go_to_process():
    print('Processing 2 clusters')
    process_cluster(clus_list2, 2)
    print('Processing 3 clusters')
    process_cluster(cluster_list3, 3)
    print('Processing 4 clusters')
    process_cluster(cluster_list4, 4)


def get_devices_overlap():
    global OVERLAP  # divisor for scaling
    with open(dir_path_pivot) as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        for row in readCSV:
            # read pivot data
            pivot_array.append(round(float(row[2]), 3))
        b = [sum(pivot_array[i:i + OVERLAP])
             for i in range(0, STOP, STEP)]
        # print(b)

    # print(device_list)
    for f in sorted(os.listdir(dir_path)):
        if f.endswith('.csv'):
            file_path: str = dir_path + f
            # print(str(f))
            duration = []
            total: float = 0.0
            with open(file_path, 'r') as file_D:
                spam_reader = csv.reader(file_D, delimiter=',')
                for row in spam_reader:
                    duration.append(round(float(row[2]), 3))
                d = [sum(duration[i:i + OVERLAP])
                     for i in range(0, STOP, STEP)]
                # deviations from pivot
                mini_array = [round((d_i - b_i) / OVERLAP, 3)
                              for d_i, b_i in zip(d, b)]
                # add deviation for each device to a list
                global_array.append(mini_array)
                # print(mini_array)

    return transpose(global_array)
    # print(transpose_list)
    # print(len(transpose_list[1]))


def getDevices():
    global DIVISOR  # divisor for scaling
    with open(dir_path_pivot) as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        for row in readCSV:
            # read pivot data
            pivot_array.append(round(float(row[2]), 3))
        b = [sum(pivot_array[i:i + DIVISOR])
             for i in range(0, len(pivot_array), DIVISOR)]
        # print(b)

    # print(device_list)
    for f in sorted(os.listdir(dir_path)):
        if f.endswith('.csv'):
            file_path: str = dir_path + f
            # print(str(f))
            duration = []
            total: float = 0.0
            with open(file_path, 'r') as file_D:
                spam_reader = csv.reader(file_D, delimiter=',')
                for row in spam_reader:
                    duration.append(round(float(row[2]), 3))
                d = [sum(duration[i:i + DIVISOR])
                     for i in range(0, len(duration), DIVISOR)]
                # deviations from pivot
                mini_array = [round((d_i - b_i) / DIVISOR, 3)
                              for d_i, b_i in zip(d, b)]
                # add deviation for each device to a list
                global_array.append(mini_array)
                # print(mini_array)

    return transpose(global_array)
    # print(transpose_list)
    # print(len(transpose_list[1]))


def k_Means(y):
    # Specify the number of clusters (3) and fit the data X
    x, out = detectOutliers(y)

    # x = X

    k_means = KMeans(n_clusters=3, random_state=5, init='k-means++').fit(x)

    # Get the cluster centroids
    labels = k_means.labels_
    # cluster_centers = k_means.cluster_centers_

    # Get the cluster labels
    labels_unique = np.unique(labels)
    n_clusters_ = len(labels_unique)
    cluster_list.append(labels)
    print("===================K-Means==================")
    first = []
    for k in range(n_clusters_):
        my_members = labels == k
        print("cluster {0}: {1} ==> {2}".format(k + 1, x[my_members, 0],
                                                len(x[my_members, 0])))
        print("cluster {0}: {1} ==> {2}".format(k + 1, np_dev_list[my_members, 0],
                                                len(np_dev_list[my_members, 0])))
        first.append(len(x[my_members, 0]))
    first.append(out)
    second.append(first)

    # Plotting the cluster centers and the data points on a 2D plane
    # plt.scatter(X[:, 0], X[:, -1])
    #
    # plt.scatter(k_means.cluster_centers_[:, 0],
    #             k_means.cluster_centers_[:, 1], c='red', marker='x')

    # plt.title('Data points and cluster centroids')
    # plt.show()


def dbscan(x):
    global two_count
    global three_count
    global four_count
    global five_count

    global MIN_SAMPLES  # dbscan min samples
    global EPS  # dbscan epsilon value
    global DIVISOR  # divisor for scaling

    db_default = DBSCAN(min_samples=MIN_SAMPLES, eps=EPS).fit(x)
    labels = db_default.labels_
    # clusters = db_default.fit_predict(X)
    labels_unique = np.unique(labels)
    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)
    print("===================DBSCAN==================")
    num_clusters.append(n_clusters_)
    # print(labels)
    print('Estimated number of clusters: %d' % n_clusters_)
    print('Estimated number of noise points: %d' % n_noise_)

    # plot the cluster assignments
    # plt.scatter(X[:, 0], X[:, 1], c=clusters, cmap="plasma")
    # plt.xlabel("Normalized deviation")
    # second = []
    # third = []
    # fourth = []
    twos = []
    threes = []
    fours = []
    outliers = []
    outlier = []
    # plt.show()
    cluster_list.append(labels)

    # if n_clusters_ == 1:
    #     for k in range(len(labels_unique)):
    #         my_members = labels == k
    #         print("cluster {0}: {1} ==> {2}".format(k, x[my_members, 0],
    #                                                 len(x[my_members, 0])))
    #         print("cluster {0}: {1} ==> {2}".format(k, np_dev_list[my_members,0],
    #                                                 len(np_dev_list[my_members, 0])))

    if n_clusters_ == 2:
        clus_list2.append(labels)
        two_count += 1
        mean2 = []
        # for k in range(-1, len(labels_unique) - 1, 1):
        for k in range(len(labels_unique)):
            my_members = labels == k
            twos.append(len(np_dev_list[my_members, 0]))
            second_ = x[my_members, 0].tolist()
            if len(second_) > 0:
                val2 = np.max(second_) - np.min(second_)
                mean2.append(np.mean(second_))
            print("cluster {0}: {1} ==> {2}".format(k, x[my_members, 0],
                                                    len(x[my_members, 0])))
            print("cluster {0}: {1} ==> {2}".format(k, np_dev_list[my_members, 0],
                                                    len(np_dev_list[my_members, 0])))

        twos.sort()
        if twos[1] < 5:
            twos.clear()
        else:
            twos.append(n_noise_)
        range2.append(val2)
        my_list2 = [np.abs(m - n) for m in mean2 for n in mean2 if m != n]
        for row in my_list2:
            average2.append(row)
    elif n_clusters_ == 3:
        clus_list3.append(labels)
        three_count += 1
        mean3 = []
        for k in range(len(labels_unique)):
            my_members = labels == k
            threes.append(len(np_dev_list[my_members, 0]))
            third_ = x[my_members, 0].tolist()
            if len(third_) > 0:
                val3 = np.max(third_) - np.min(third_)
                mean3.append(np.mean(third_))
            print("cluster {0}: {1} ==> {2}".format(k, x[my_members, 0],
                                                    len(x[my_members, 0])))
            print("cluster {0}: {1} ==> {2}".format(k, np_dev_list[my_members, 0],
                                                    len(np_dev_list[my_members, 0])))
        threes.sort()
        threes.append(n_noise_)
        range3.append(val3)
        my_list3 = [np.abs(m - n) for m in mean3 for n in mean3 if m != n]
        for row in my_list3:
            average3.append(row)
    elif n_clusters_ == 4:
        clus_list4.append(labels)
        four_count += 1
        mean4 = []
        for k in range(len(labels_unique)):
            my_members = labels == k
            fours.append(len(np_dev_list[my_members, 0]))
            fourth_ = x[my_members, 0].tolist()

            # print(fourth)
            # print(type(fourth))
            if len(fourth_) > 0:
                val4 = np.max(fourth_) - np.min(fourth_)
                mean4.append(np.mean(fourth_))
            print("cluster {0}: {1} ==> {2}".format(k, x[my_members, 0],
                                                    len(x[my_members, 0])))
            print("cluster {0}: {1} ==> {2}".format(k, np_dev_list[my_members, 0],
                                                    len(np_dev_list[my_members, 0])))
        fours.sort()
        fours.append(n_noise_)
        range4.append(val4)
        my_list4 = [np.abs(m - n) for m in mean4 for n in mean4 if m != n]
        for row in my_list4:
            average4.append(row)
        # first.append(len(np_dev_list[my_members, 0]))
    # second.append(first)
    if len(twos) > 0:
        big_twos_.append(twos)
    if len(threes) > 0:
        big_threes_.append(threes)
    if len(fours) > 0:
        big_fours_.append(fours)


def mean_shift(x):
    global two_count
    global three_count
    global four_count
    global five_count
    # x, out = detectOutliers(X)
    # The following bandwidth can be automatically detected using
    bandwidth = estimate_bandwidth(x, quantile=0.55, n_samples=35)

    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    ms.fit(x)
    labels = ms.labels_
    cluster_centers = ms.cluster_centers_
    labels_unique = np.unique(labels)
    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)
    labels_unique = np.unique(labels)
    # n_clusters_ = len(labels_unique)

    first, twos, threes, fours = [], [], [], []
    outliers = []
    cluster_list.append(labels)
    print("===================Mean Shift==================")
    num_clusters.append(n_clusters_)
    if n_clusters_ == 2:
        cluster_list2.append(labels)
        two_count += 1
        for k in range(len(labels_unique)):
            my_members = labels == k
            twos.append(len(np_dev_list[my_members, 0]))

            print("cluster {0}: {1} ==> {2}".format(k, x[my_members, 0],
                                                    len(x[my_members, 0])))
            print("cluster {0}: {1} ==> {2}".format(k, np_dev_list[my_members, 0],
                                                    len(np_dev_list[my_members, 0])))
    elif n_clusters_ == 3:
        cluster_list3.append(labels)
        three_count += 1
        for k in range(-1, len(labels_unique) - 1, 1):
            my_members = labels == k
            threes.append(len(np_dev_list[my_members, 0]))

            # print("cluster {0}: {1} ==> {2}".format(k, x[my_members, 0],
            #                                         len(x[my_members, 0])))
            # print("cluster {0}: {1} ==> {2}".format(k, np_dev_list[my_members, 0],
            #                                         len(np_dev_list[my_members, 0])))
    elif n_clusters_ == 4:
        cluster_list4.append(labels)
        four_count += 1
        for k in range(-1, len(labels_unique) - 1, 1):
            my_members = labels == k
            fours.append(len(np_dev_list[my_members, 0]))

            # print("cluster {0}: {1} ==> {2}".format(k, x[my_members, 0],
            #                                         len(x[my_members, 0])))
            # print("cluster {0}: {1} ==> {2}".format(k, np_dev_list[my_members, 0],
            #                                         len(np_dev_list[my_members, 0])))
        # first.append(len(np_dev_list[my_members, 0]))
    # second.append(first)
    if len(twos) > 0:
        big_twos_.append(twos)
    if len(threes) > 0:
        big_threes_.append(threes)
    if len(fours) > 0:
        big_fours_.append(fours)


def affinity_propagation(x):
    af = AffinityPropagation(damping=0.93).fit(x)
    cluster_centers_indices = af.cluster_centers_indices_
    labels = af.labels_

    n_clusters_ = len(cluster_centers_indices)

    cluster_list.append(labels)
    print("===================Affinity Propagation==================")
    num_clusters.append(n_clusters_)
    for k in range(n_clusters_):
        my_members = labels == k
        print("cluster {0}: {1} ==> {2}".format(k + 1, x[my_members, 0],
                                                len(x[my_members, 0])))

    print('Number of clusters: %d' % n_clusters_)


def process_cluster_all_to_all(list_, k):
    similarity_data = []
    v_measure_data = []

    print(len(list_))
    for m in range(len(list_)):
        sim_list = []
        v_list = []
        sim_average = []
        v_average = []
        # print("Clustering point: {0}".format(m + 1))
        for n in range(len(list_)):
            if m != n:
                similarity_score = metrics.adjusted_rand_score(
                    list_[m], list_[n])
                v_measure_score = metrics.v_measure_score(
                    list_[m], list_[n], beta=1.8)

                sim_list.append(similarity_score)
                v_list.append(v_measure_score)

        # print("Similarity Score:")
        # seed(4)
        # sim_list = [x - (random() * .16) for x in sim_list]
        # print(sim_list)
        # if np.mean(sim_list) > 0.6 and k == 2:
        if np.mean(sim_list) > 0.8 and k == 2:
            similarity_data.append(np.mean(sim_list))
            v_measure_data.append(v_list)
        # if np.mean(sim_list) > 0.35 and k == 3:
        if np.mean(sim_list) > 0.6 and k == 3:
            similarity_data.append(np.mean(sim_list))
            v_measure_data.append(v_list)
        if np.mean(sim_list) > 0.5 and k == 4:
            similarity_data.append(np.mean(sim_list))
            v_measure_data.append(v_list)

        # print("V_measure Score:")
        # seed(1)
        # v_list = [x - (random() * .16) for x in v_list]
        # print(v_list)

        sim_average = [np.mean(x) for x in similarity_data]
        # print(sim_average)
        v_average = [np.mean(x) for x in v_measure_data]
        # print(v_average)
    combined_list_to_plot.append(similarity_data)
    boxPlot(similarity_data, v_measure_data, k)

def process_cluster(_list_, k):
    similarity_data_ = []
    v_measure_data_ = []

    print(len(_list_))
    pivot_compare = _list_[0]
    for m in range(len(_list_)):
        sim_list = []
        v_list = []
        sim_average = []
        v_average = []

        print("Clustering point: {0}".format(m + 1))
        similarity_score = metrics.adjusted_rand_score(
            _list_[m], pivot_compare)
        v_measure_score = metrics.v_measure_score(
            _list_[m], pivot_compare, beta=1.8)

        sim_list.append(similarity_score)
        v_list.append(v_measure_score)

        print("Similarity Score:")
        # seed(4)
        # sim_list = [x - (random() * .16) for x in sim_list]
        # print(sim_list)
        # if np.mean(sim_list) > 0.6 and k == 2:
        if k == 2:
            similarity_data_.append(sim_list)
            v_measure_data_.append(v_list)
        # if np.mean(sim_list) > 0.35 and k == 3:
        if k == 3:
            similarity_data_.append(sim_list)
            v_measure_data_.append(v_list)
        if k == 4:
            similarity_data_.append(sim_list)
            v_measure_data_.append(v_list)

        # print("V_measure Score:")
        # seed(1)
        # v_list = [x - (random() * .16) for x in v_list]
        # print(v_list)

        sim_average = [np.mean(x) for x in similarity_data_]
        # print(sim_average)
        v_average = [np.mean(x) for x in v_measure_data_]
        # print(v_average)

    boxPlot(similarity_data_, v_measure_data_, k)

def boxPlot(sim_data, v_data, k):
    fig = plt.figure()
    # plt.boxplot(sim_data, patch_artist=True,
    #             labels=['Pt1', 'Pt2', 'Pt3',
    #                     'Pt4', 'Pt5', 'Pt6',
    #                     'Pt7', 'Pt8', 'Pt9'])
    plt.boxplot(sim_data, patch_artist=True)
    plt.xlabel('Clustering point', fontsize=15, fontweight='bold')
    plt.ylabel('Rand index score', fontsize=15, fontweight='bold')
    plt.tick_params(axis='x', labelsize=15)
    plt.tick_params(axis='y', labelsize=15)
    plt.title('{0} Clusters/clustering point'.format(k), fontsize=15, fontweight='bold')
    plt.figure(figsize=(3.5, 4.5))
    plt.rc('font', **font)
    #fig.savefig('Results/sim_dbscan_new{0}_new_june6.pdf'.format(k))
    plt.show()

    # fig = plt.figure()
    # # plt.boxplot(v_data, patch_artist=True,
    # #             labels=['Pt1', 'Pt2', 'Pt3',
    # #                     'Pt4', 'Pt5', 'Pt6',
    # #                     'Pt7', 'Pt8', 'Pt9'])
    # plt.boxplot(v_data, patch_artist=True)
    # plt.xlabel('Clustering point')
    # plt.ylabel('V-measure')
    # plt.title('{0} Clusters/clustering point'.format(k))
    # # fig.savefig('Results/measure_kmeans{0}.pdf'.format(k))
    # plt.show()


def checker():
    #print(combined_list_to_plot)
    # print(len(combined_list_to_plot))
    #
    # fig = plt.figure()
    # # plt.boxplot(sim_data, patch_artist=True,
    # #             labels=['Pt1', 'Pt2', 'Pt3',
    # #                     'Pt4', 'Pt5', 'Pt6',
    # #                     'Pt7', 'Pt8', 'Pt9'])
    # plt.boxplot(combined_list_to_plot, patch_artist=True,
    #             labels=['2-Clusters', '3-Clusters', '4-Clusters'])
    # #plt.xlabel('Number of clusters per point', fontsize=15, fontweight='bold')
    # plt.ylabel('Rand index score', fontsize=14, fontweight='bold')
    # plt.tick_params(axis='x', labelsize=14)
    # plt.tick_params(axis='y', labelsize=14)
    # #plt.title('{0} Clusters/clustering point'.format(k), fontsize=15, fontweight='bold')
    # #plt.figure(figsize=(4.0, 4.8))
    # fig.savefig('Results/sim_dbscan_all_points.pdf')
    # plt.show()

    range2_mean_ = np.mean(combined_list_to_plot[0])
    range3_mean_ = np.mean(combined_list_to_plot[1])
    range4_mean_ = np.mean(combined_list_to_plot[2])

    range2_std_ = np.std(combined_list_to_plot[0])
    range3_std_ = np.std(combined_list_to_plot[1])
    range4_std_ = np.std(combined_list_to_plot[2])


    # print(range2_mean_)
    # print(range3_mean_)
    # print(range4_mean_)

    # Create lists for the plot
    labels = ['2', '3', '4']
    x_pos = np.arange(len(labels))
    CTEs_range = [range2_mean_, range3_mean_, range4_mean_]
    error_range = [range2_std_, range3_std_, range4_std_]
    width = 0.3

    # Build the plot
    fig, ax = plt.subplots()
    p1 = ax.bar(x_pos, CTEs_range, width=width, yerr=error_range,
                align='center', capsize=10, color='lightblue',
                hatch = 'o', edgecolor = 'black')

    ax.set_xlabel('Clusters per clustering point', fontsize=18)
    ax.set_ylabel('Rand index score', fontsize=18)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, fontsize=18)
    ax.tick_params(axis='y', labelsize=16)
    #ax.legend((p1[0]), ('Intra-cluster distance', 'Inter-cluster distance'),
              #fontsize=18)
    # ax.yaxis.grid(True)

    # Save the figure and show
    plt.tight_layout()
    plt.rc('font', **font)
    plt.savefig('Results/sim_dbscan_all_points_new_sept16.pdf')
    plt.show()


def plotHisto():
    labels = ['Pt1', 'Pt2', 'Pt3',
              'Pt4', 'Pt5', 'Pt6']

    listToPlot = transpose(second)
    first_ = listToPlot[0]
    second_ = listToPlot[1]
    third_ = listToPlot[2]
    fourth_ = listToPlot[3]
    out_ = listToPlot[4]

    x = np.arange(len(labels))  # the label locations
    width = 0.15  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - 2 * width, first_, width, label='C1')
    rects2 = ax.bar(x - width, second_, width, label='C2')
    rects3 = ax.bar(x, third_, width, label='C3')
    rects4 = ax.bar(x + width, fourth_, width, label='C4')
    rects5 = ax.bar(x + 2 * width, out_, width, label='OUT')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Number of devices')
    ax.set_title('Number of Devices per Cluster (4 Clusters)')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(bbox_to_anchor=(1.04, 1), loc=2, borderaxespad=0.)

    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)
    autolabel(rects4)
    autolabel(rects5)

    fig.tight_layout()
    #fig.savefig('devices4.pdf')
    plt.show()


def plot_histo_cumm(k):
    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    if k == 2:
        labels = ['2-Clusters', '3-Clusters', '4-clusters']

        to_plot = transpose(second)
        first_ = to_plot[0]
        second_ = to_plot[1]
        third_ = to_plot[2]
        fourth_ = to_plot[3]
        out_ = to_plot[4]

        x = np.arange(len(labels))  # the label locations
        width = 0.15  # the width of the bars

        fig, ax = plt.subplots()
        rects1 = ax.bar(x - 2 * width, first_, width, label='C1')
        rects2 = ax.bar(x - width, second_, width, label='C2')
        rects3 = ax.bar(x, third_, width, label='C3')
        rects4 = ax.bar(x + width, fourth_, width, label='C4')
        rects5 = ax.bar(x + 2 * width, out_, width, label='OUT')

        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_ylabel('Number of devices')
        ax.set_title('Number of Devices per Cluster (4 Clusters)')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend(bbox_to_anchor=(1.04, 1), loc=2, borderaxespad=0.)

        autolabel(rects1)
        autolabel(rects2)
        autolabel(rects3)
        autolabel(rects4)
        autolabel(rects5)

        fig.tight_layout()
        fig.savefig('devices4.pdf')
        plt.show()


def transpose(t):
    # convert list to array
    t1 = np.asarray(t)
    # transpose array to be clustered
    t2 = t1.T
    # convert back to list
    t3 = t2.tolist()
    return t3


def detectOutliers(t):
    t3_1, t3_2 = zip(*t)
    t3_list = list(t3_1)
    t1 = []
    i: int = 0
    q1 = np.percentile(t3_list, 25, interpolation='midpoint')
    q3 = np.percentile(t3_list, 75, interpolation='midpoint')
    iqr = q3 - q1
    upper, lower = q3 + 2.0 * iqr, q1 - 2.0 * iqr

    for t2 in t3_list:
        if upper > t2 > lower:
            t1.append(t2)
        else:
            i = i + 1
    return np.asarray(list(zip(t1, np.zeros(len(t1))))), i


main()

