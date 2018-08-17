'''
	This utility file contains methods to convert the relationship file
	generated from the MySQL queries to the format Graph convolutional network expects
'''
import pandas as pd
import json
from sklearn.model_selection import train_test_split
import numpy as np
import pickle as pkl
import scipy
'''
	genereates one hot labels for each label in the file
'''
def get_one_hot_labels(label_file_name):
    data = pd.read_csv(label_file_name, sep=" ", header=None)
    data.columns = ["id", "label"]
    one_hot = pd.get_dummies(data["label"])
    data.drop('label', axis=1, inplace=True)
    return data.join(one_hot).sort_values('id').reset_index(drop=True)
'''
	generates the embedding file which is generated from either MySQL queries based on the orientation,shape or
	generated from unsupervised model like deep walk.
'''
def get_embedding_df(emedding_file_name):
    data = pd.read_csv(emedding_file_name, sep=" ", header=None)
    data.sort_values(data.columns[0]).reset_index(drop=True)
    return data
def search(element, list_):
    if element in list_:
        return True
    return False
'''
	generates the JSON file which is in the format for GraphSAGE algorithm.
	The method splits the data into training, test and validation sets
	Also since the inductive nature of the algorithm, we drop the edges of nodes which are present in the test and validation set.
	The resultant file is persisted to multiple file with the prefix of the file being consistent across all the output file in all the files
'''
def generate_G_Json(label_file_name, embedding_file_name, edges_file_name):
    embedding = get_embedding_df(embedding_file_name)
    label = get_one_hot_labels(label_file_name)
    
    train_test = train_test_split(np.arange(0, label.shape[0]), shuffle=True,  test_size=0.2)
    train = map(int, train_test[0].tolist())
    test = map(int, train_test[1].tolist())
    data = {}
    edges_file = open(edges_file_name, 'r')
    for edge in edges_file:
        nodes = edge.split(' ')
        if int(nodes[0]) in data:
            data[int(nodes[0])].append(int(nodes[1]))
        else:
            data[int(nodes[0])] = [int(nodes[1])]
    assert set(embedding[embedding.columns[0]].values.tolist()) == set(embedding.index)
    x_ = embedding[embedding.index.isin(train)]
    tx_ = embedding[embedding.index.isin(test)]
    allx_ = embedding
    y_ = label[label['id'].isin(train)]
    ty_ = label[label['id'].isin(test)]
    ally_ = label
    assert len(set(x_[x_.columns[0]].values.tolist()).intersection(tx_[tx_.columns[0]].values.tolist())) == 0
    assert len(set(y_['id'].values.tolist()).intersection(ty_['id'].values.tolist())) == 0
    x_output = open('GCN/data/ind.gds.x', 'wb')
    tx_output = open('GCN/data/ind.gds.tx', 'wb')
    allx_output = open('GCN/data/ind.gds.allx', 'wb')
    y_output = open('GCN/data/ind.gds.y', 'wb')
    ty_output = open('GCN/data/ind.gds.ty', 'wb')
    ally_output = open('GCN/data/ind.gds.ally', 'wb')
    graph_output = open('GCN/data/ind.gds.graph', 'wb')
    test_index_output = open('GCN/data/ind.gds.test.index', 'w')

    pkl.dump(scipy.sparse.csr_matrix(x_.values), x_output)
    pkl.dump(scipy.sparse.csr_matrix(tx_.values), tx_output)
    pkl.dump(scipy.sparse.csr_matrix(x_.values), allx_output)
    pkl.dump(y_.values, y_output)
    pkl.dump(ty_.values, ty_output)
    pkl.dump(y_.values, ally_output)
    pkl.dump(data, graph_output)
    for item in test:
        test_index_output.write("%s\n" %item) 
generate_G_Json('data/node_labels_final_reordered.txt',
               'data/node_features_length_reordered.txt',
                   'data/relationship_final_mysql_nodes_reordered.txt')
