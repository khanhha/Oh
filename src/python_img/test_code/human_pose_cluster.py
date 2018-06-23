import os
from time import time
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import operator
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.cm as cm

def select_best_cluster(data):
    range_n_clusters = np.arange(10, 40, 1)
    cluster_scores = {}
    for n_clusters in range_n_clusters:
        # Initialize the clusterer with n_clusters value and a random generator
        # seed of 10 for reproducibility.
        clusterer = KMeans(n_clusters=n_clusters, random_state=n_clusters)
        cluster_labels = clusterer.fit_predict(data)

        # The silhouette_score gives the average value for all the samples.
        # This gives a perspective into the density and separation of the formed
        # clusters
        silhouette_avg = silhouette_score(data, cluster_labels)
        print("For n_clusters =", n_clusters,
              "The average silhouette_score is :", silhouette_avg)

        cluster_scores[n_clusters] = silhouette_avg
        continue

        # Create a subplot with 1 row and 2 columns
        fig, (ax1) = plt.subplots(1, 1)
        fig.set_size_inches(18, 7)

        # The 1st subplot is the silhouette plot
        # The silhouette coefficient can range from -1, 1 but in this example all
        # lie within [-0.1, 1]
        ax1.set_xlim([-0.1, 1])
        # The (n_clusters+1)*10 is for inserting blank space between silhouette
        # plots of individual clusters, to demarcate them clearly.
        ax1.set_ylim([0, len(data) + (n_clusters + 1) * 10])

        # Compute the silhouette scores for each sample
        sample_silhouette_values = silhouette_samples(data, cluster_labels)

        y_lower = 10
        for i in range(n_clusters):
            # Aggregate the silhouette scores for samples belonging to
            # cluster i, and sort them
            ith_cluster_silhouette_values = \
                sample_silhouette_values[cluster_labels == i]

            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            color = cm.spectral(float(i) / n_clusters)
            ax1.fill_betweenx(np.arange(y_lower, y_upper),
                              0, ith_cluster_silhouette_values,
                              facecolor=color, edgecolor=color, alpha=0.7)

            # Label the silhouette plots with their cluster numbers at the middle
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples

        ax1.set_title("The silhouette plot for the various clusters.")
        ax1.set_xlabel("The silhouette coefficient values")
        ax1.set_ylabel("Cluster label")

        # The vertical line for average silhouette score of all the values
        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")
        ax1.set_yticks([])  # Clear the yaxis labels / ticks
        ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

        plt.savefig('D:\\Projects\\Oh\\data\\images\\prid_450s\\cluster_measure\\' + str(n_clusters) + '.png')

    best_cluster = max(cluster_scores.items(), key=lambda x: x[1])[0]
    print('best cluster number = ', best_cluster, ', Score = ', cluster_scores[best_cluster])
    return best_cluster, cluster_scores[best_cluster]

dir_path = 'D:\Projects\Oh\data\images\prid_450s\cam_a_silhouette'
entries = os.listdir(dir_path)
imgs = list()
img_names = list()
for entry in entries:
    path = os.path.join(dir_path, entry)
    img = cv.imread(path)
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    imgs.append(img.flatten())
    img_names.append(entry)

data = np.vstack(imgs)

#best_cluster_number, score = select_best_cluster(data)
best_cluster_number = 11; score = 0.04

print('result: ', best_cluster_number, ' ', score)

estimator = KMeans(init='k-means++', n_clusters= best_cluster_number, n_init=best_cluster_number)
estimator.fit(data)
uni_labels = np.unique(estimator.labels_)
for l in uni_labels:
    dir = 'D:\\Projects\\Oh\\data\images\\prid_450s\\cluster_images\\'+str(l) + '\\'
    try:
        os.stat(dir)
    except:
        os.mkdir(dir)
    for i in range(len(estimator.labels_)):
        if estimator.labels_[i]== l:
            img = data[i].reshape(128, 64)
            file_path = dir + img_names[i]
            cv.imwrite(file_path, img)

label_file_path = 'D:\\Projects\\Oh\\data\images\\prid_450s\\cam_a_label.txt'
file = open(label_file_path, 'w')
#path_lables = list()
for i in range(len(estimator.labels_)):
    name = img_names[i].replace('man','img')
    file_path = os.path.join('D:\Projects\Oh\data\images\prid_450s\cam_a', name)
    file.write(file_path + ' ' + str(estimator.labels_[i]) + '\n')
    #path_lables.append(file_path + ' ' + str(estimator.labels_[i]) + '\n')
file.close()





