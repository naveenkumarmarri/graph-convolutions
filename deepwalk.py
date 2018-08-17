import sys
import os
import subprocess
import numpy as np
import tensorflow as tf
from operator import itemgetter
from sklearn.model_selection import train_test_split
os.chdir('/home/dataset/gds/final/nodes_new_api/')
'''
	This file contains for invoking deepwalk and performing classfication
	using a deep neural network on the representation learnt from the
	deep walk algorithm.
	The deep walk algorithm code can be found on https://github.com/phanein/deepwalk
'''
subprocess.call("deepwalk --input relationship_final_mysql_nodes_reordered.txt --format edgelist \
--number-walks 100 --representation-size 256 --walk-length 20 --window-size 10\
 --workers 10 --output data/gds_graph.embeddings", shell=True)  

'''
	Utility function to sort the input dictionary based on the key
'''
def sort_labels(labels):
    sorted(labels, key=itemgetter(0))
    return labels
'''
	The following code is for reading the labels file which sorts
	the labels based on the id of the node because we need to make the order
	of the embedding file and the order of the labels same
'''
def read_labels_file(file_name):
    f = open(file_name, 'r')
    distinct_labels = []
    labels = []
    for line in f:
        node,label = line.split(" ")
        labels.append([int(node), label.rstrip()])
    for label in labels:
        if label[1] not in distinct_labels:
            distinct_labels.append(label[1])
    sorted_labels = sort_labels(labels)
    out_labels = []
    for label in sorted_labels:
        label[1] = distinct_labels.index(label[1])
    for label in sorted_labels:
        out_labels.append(label[1])
    return out_labels, distinct_labels
'''
	The following helper method reads the embedding file and sorts the node if
	based on the id of the node which helps to keep the embedding order and 
	label order in sync
'''
def read_embeddings_file(file_name):
    f = open(file_name, 'r')
    next(f)
    features = []
    values = {}
    for line in f:
        tokens = line.split(" ")
        if tokens[0] not in values:
            values[int(tokens[0])] = tokens[1:]
        else:
            print("found duplicate")
    values.keys().sort()
    keys = values.keys()
    features = []
    for key in keys:
        features.append(values[key])
    return features
features = read_embeddings_file('gds_graph.embeddings')
# labels, label_alias = read_labels_file('/home/dataset/gds/final/gds_graph_final_labels.txt')
labels, label_alias = read_labels_file('node_labels_final.txt')
input_features = np.array(features, dtype=float)
output_features = np.array(labels, dtype=int)
one_hot_encoded_labels = np.zeros((output_features.size, output_features.max()+1))
one_hot_encoded_labels[np.arange(output_features.size),output_features] = 1
one_hot_encoded_labels = one_hot_encoded_labels.astype(int)
X_train, X_test, y_train, y_test = train_test_split(input_features, one_hot_encoded_labels,\
                                                    test_size=0.2)
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="3"
tf.reset_default_graph()
num_iterations = 20
num_steps = input_features.shape[0] / 10
learning_rate = 0.001

# class_weight = tf.constant([[(freq.sum() - freq[0]) / float(freq.sum()), \
#                              (freq.sum() - freq[1]) / float(freq.sum()), \
#                              (freq.sum() - freq[2]) / float(freq.sum()) ]])

X = tf.placeholder(tf.float32, shape=[None, X_train.shape[1]])
y = tf.placeholder(tf.float32, shape=[None, y_train.shape[1]])
prob = tf.placeholder_with_default(1.0, shape=())


# weight_per_label = tf.transpose( tf.matmul(y
#                            , tf.transpose(class_weight)) ) 

layer1 = tf.layers.dense(X, 200, activation=tf.nn.sigmoid, kernel_initializer=tf.keras.initializers.he_normal())
dropout_1 = tf.nn.dropout(layer1, prob)

layer2 = tf.layers.dense(dropout_1, 150, activation=tf.nn.sigmoid, kernel_initializer=tf.keras.initializers.he_normal())
dropout_2 = tf.nn.dropout(layer2, prob)
layer3 = tf.layers.dense(dropout_2, 100, activation=tf.nn.sigmoid, kernel_initializer=tf.keras.initializers.he_normal())
dropout_3 = tf.nn.dropout(layer3, prob)
layer4 = tf.layers.dense(dropout_3, 50, activation=tf.nn.sigmoid, kernel_initializer=tf.keras.initializers.he_normal())
dropout_4 = tf.nn.dropout(layer4, prob)
layer5 = tf.layers.dense(dropout_4, 25, activation=None, kernel_initializer=tf.keras.initializers.he_normal())

              
y_ = tf.nn.softmax(layer5)
loss_op = tf.nn.softmax_cross_entropy_with_logits_v2(logits=y_,
                                         labels = y)

#adam optimizier as the optimization function
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

#extract the correct predictions and compute the accuracy
correct_pred = tf.equal(tf.argmax(y_, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

#initialize all the variables
init = tf.global_variables_initializer()

# Start training
with tf.Session() as sess:
    sess.run(init)
    for i in range(0, num_iterations):
        acc_ = []
        for step in range(0, num_steps):
            batch_x = input_features[step:step+10]
            batch_y = one_hot_encoded_labels[step:step+10]
            _, acc = sess.run([train_op,accuracy], feed_dict={X: batch_x, y: batch_y,prob: 0.1})
            acc_.append(acc)
        print("The accuracy at iteration ", i , "is ", np.array(acc_).mean())
    test_accuracy = sess.run([ accuracy], feed_dict={X: X_test, y:y_test})
    print("The test accuracy is ", test_accuracy)

