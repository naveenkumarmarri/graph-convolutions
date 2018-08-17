from __future__ import print_function
import os

import tensorflow as tf
import csv
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import matplotlib
from operator import itemgetter

import networkx as nx
from networkx.algorithms.bipartite import biadjacency_matrix

%matplotlib inline

# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"]= "4"

config = tf.ConfigProto()
config.gpu_options.allow_growth=True
'''
	converting a bipartite graph in adjacency matrix
'''
def to_adjacency_matrix(data):
    g = nx.DiGraph()
    g.add_edges_from(data)
    partition_1 = set(map(itemgetter(0), data))
    return partition_1, biadjacency_matrix(g, partition_1).toarray()

#Input file which stores the bipartite graph information
f = open('data/relationship_final_mysql_nodes_reoredered.txt')
edges = []
for line in f:
    tokens = line.split(' ')
    edges.append(tokens)
partition, adj_matrix = to_adjacency_matrix(edges)	#Computes the adjacency matrix from a bipatite graph
S = cosine_similarity(adj_matrix)			#computes cosine similarity between each node in the graph
D = np.diag((np.sum(adj_matrix, axis=1)))		#Computes the degree of each node in the graph and diagonalizes a matrix
inp = np.matmul(np.linalg.inv(D), S)
from tensorflow.python.framework import ops
ops.reset_default_graph()
learning_rate = 0.001
p = tf.constant(0.01)
beta = 0.01

X = tf.placeholder(tf.float32, [None, inp.shape[1]])
Y = tf.placeholder(tf.float32, [None, inp.shape[1]])

#Encoding layers
layer1 = tf.layers.dense(X, 42922, activation=tf.nn.sigmoid, kernel_initializer=tf.keras.initializers.he_normal())
layer2 = tf.layers.dense(layer1, 20000, activation=tf.nn.sigmoid, kernel_initializer=tf.keras.initializers.he_normal())
layer3 = tf.layers.dense(layer2, 10000, activation=tf.nn.sigmoid, kernel_initializer=tf.keras.initializers.he_normal())
layer4 = tf.layers.dense(layer2, 5000, activation=tf.nn.sigmoid, kernel_initializer=tf.keras.initializers.he_normal())
layer5 = tf.layers.dense(layer2, 4000, activation=tf.nn.sigmoid, kernel_initializer=tf.keras.initializers.he_normal())


#decoding layers
layer4_decode = tf.layers.dense(layer5, 5000, activation=tf.nn.sigmoid,
                                kernel_initializer=tf.keras.initializers.he_normal())
layer3_decode = tf.layers.dense(layer4_decode, 10000, activation=tf.nn.sigmoid, 
                                kernel_initializer=tf.keras.initializers.he_normal())
layer2_decode = tf.layers.dense(layer3_decode, 20000, activation=tf.nn.sigmoid, 
                                kernel_initializer=tf.keras.initializers.he_normal())
layer1_decode = tf.layers.dense(layer2_decode, 42922, activation=tf.nn.sigmoid, 
                                kernel_initializer=tf.keras.initializers.he_normal())


p_hat = tf.div(tf.reduce_sum(layer5, 0), 1362)

cost = tf.reduce_mean(tf.pow(layer1 - layer1_decode, 2))
cost_sparse = tf.multiply(0.01, tf.reduce_mean(p* tf.log(p/p_hat)+(1-p)*tf.log((1-p)/(1-p_hat))))
total_cost = tf.add(cost, cost_sparse)

optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
train_op = optimizer.minimize(total_cost)


init = tf.global_variables_initializer()

with tf.Session(config=config) as sess:
    sess.run(init)
    for i in range(0, 30):
        err_sum = []
        for step in range(0, 10):
            _, err,sae = sess.run([train_op, total_cost, layer5], feed_dict={X:inp[step:step+10], Y:inp[step:step+10]})
            err_sum.append(err) 
        print("Error at epoch", i, "is ", np.array(err_sum).mean()*100)
#Performs KMeans clustering on the hidden representation of the autoencoder
kmeans = KMeans(n_clusters=25, random_state=12319).fit(sae)
f = open('data/node_labels_final_reordered.txt')
valid_labels = {}
for line in f:
    tokens = line.split(",")
    tokens[1] = tokens[1].rstrip()
    if str(tokens[0]) not in partition:
        pass
    else:
        valid_labels[int(tokens[0])] = tokens[1]
distinct_labels = []
for k, v in valid_labels.items():
    if v not in distinct_labels:
        distinct_labels.append(v)
ordered_labels = []
for label in partition:
    temp =int(label)
    ordered_labels.append([temp, distinct_labels.index(valid_labels[temp])])
ordered_labels_list = []
for label in ordered_labels:
    ordered_labels_list.append(label[1])
pred = kmeans.labels_

df = pd.DataFrame({'Actual': ordered_labels_list, 'Predicted': pred})


ct = pd.crosstab(df['Actual'], df['Predicted'])
print(ct)


#Performing spectral clustering to compare autoencoder approach
from sklearn.cluster import SpectralClustering
from sklearn import metrics

sc = SpectralClustering(3, affinity='precomputed', n_init=100)
sc.fit(adj_matrix)

# Compare ground-truth and clustering-results
print('spectral clustering')
print(sc.labels_)
ordered_labels_list = []
for label in ordered_labels:
    ordered_labels_list.append(label[1])
pred = sc.labels_

df = pd.DataFrame({'Actual': ordered_labels_list, 'Predicted': pred})


ct = pd.crosstab(df['Actual'], df['Predicted'])
print(ct)
