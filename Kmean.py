from math import sqrt
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from csv import reader
from math import sqrt
import random
import pygame
import random
import sys
import math
from collections import deque
from queue import LifoQueue



# region KMEANS
class DataItem:
    def __init__(self, item):
        self.features = item
        self.clusterId = -1

    def getDataset(self):
        data = []
        data.append(DataItem([0, 0, 0, 0]))
        data.append(DataItem([0, 0, 0, 1]))
        data.append(DataItem([0, 0, 1, 0]))
        data.append(DataItem([0, 0, 1, 1]))
        data.append(DataItem([0, 1, 0, 0]))
        data.append(DataItem([0, 1, 0, 1]))
        data.append(DataItem([0, 1, 1, 0]))
        data.append(DataItem([0, 1, 1, 1]))
        data.append(DataItem([1, 0, 0, 0]))
        data.append(DataItem([1, 0, 0, 1]))
        data.append(DataItem([1, 0, 1, 0]))
        data.append(DataItem([1, 0, 1, 1]))
        data.append(DataItem([1, 1, 0, 0]))
        data.append(DataItem([1, 1, 0, 1]))
        data.append(DataItem([1, 1, 1, 0]))
        data.append(DataItem([1, 1, 1, 1]))
        return data

class Cluster:
        def __init__(self, id, centroid):
            self.centroid = centroid
            self.data = []
            self.id = id

        def update(self, clusterData):
            self.data = []
            for item in clusterData:
                self.data.append(item.features)
            tmpC = np.average(self.data, axis=0)
            tmpL = []
            for i in tmpC:
                tmpL.append(i)
            self.centroid = tmpL

class SimilarityDistance:
        def euclidean_distance(self, p1, p2):
            sum = 0
            for i in range(len(p1)):
                sum += (p1[i] - p2[i]) ** 2
            return sqrt(sum)

        def Manhattan_distance(self, p1, p2):
            sum = 0
            for i in range(len(p1)):
                sum += abs(p1[i] - p2[i])
            return sum

class Clustering_kmeans:
        def __init__(self, data, k, noOfIterations, isEuclidean):
            self.data = data
            self.k = k
            self.distance = SimilarityDistance()
            self.noOfIterations = noOfIterations
            self.isEuclidean = isEuclidean

        def initClusters(self):
            self.clusters = []
            for i in range(self.k):
                self.clusters.append(Cluster(i, self.data[i * 10].features))

        def getClusters(self):
            self.initClusters()


            for i in range(self.noOfIterations):
                for item in self.data:
                    min_dis = 1000

                    for cluster in self.clusters:
                        if (self.isEuclidean == 1):

                            clus_distance = self.distance.euclidean_distance(cluster.centroid, item.features)
                            if (clus_distance < min_dis):
                                item.clusterId = cluster.id
                                min_dis = clus_distance

                        else:
                            clus_distance = self.distance.Manhattan_distance(cluster.centroid, item.features)
                            if (clus_distance < min_dis):
                                item.clusterId = cluster.id
                                min_dis = clus_distance
                    cluster_items = [x for x in self.data if x.clusterId == item.clusterId]
                    self.clusters[item.clusterId].update( cluster_items)
            return self.clusters


# endregion


def Kmeans_Main():
    dataset = DataItem.getDataset(None)
    # 1 for Euclidean and 0 for Manhatan
    clustering = Clustering_kmeans(dataset, 2, len(dataset),1)
    clusters = clustering.getClusters()
    for cluster in clusters:
        for i in range(4):
            cluster.centroid[i] = round(cluster.centroid[i], 2)
        print(cluster.centroid[:4])
    return clusters

# endregion


######################## MAIN ###########################33
if __name__ == '__main__':

    Kmeans_Main()