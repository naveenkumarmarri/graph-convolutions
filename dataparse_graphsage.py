'''
	This utility file contains methods to convert the relationship file
	generated from the MySQL queries to the format GraphSAGE expects
'''
import pandas as pd
import json
from sklearn.model_selection import train_test_split
import numpy as np

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
def get_embedding_df(emedding_file_name, skip_header):
    if skip_header:
        data = pd.read_csv(emedding_file_name, sep=" ",skiprows=[0], header=None)
    else:
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
	The resultant json file is persisted to gds-G.json the prefix of the file should be consistent across all the output file in all the methods
'''
def generate_G_Json(label_file_name, embedding_file_name, edges_file_name):
    embedding = get_embedding_df(embedding_file_name)
    label = get_one_hot_labels(label_file_name)
    
    train_test = train_test_split(np.arange(0, label.shape[0]), shuffle=True,  test_size=0.2)
    train_val = train_test_split(train_test[0], shuffle=True,  test_size=0.2)
    train = map(int, train_val[0].tolist())
    val = map(int, train_val[1].tolist())
    test = map(int, train_test[1].tolist())
    data = {}
    data['directed'] = False
    data['graph'] = 'gds'
    data['multigraph'] = False
    data['nodes'] = []
    for index, row in embedding.iterrows():
        data['nodes'].append({})
        data['nodes'][index]['id'] = index
        if index in train:
            data['nodes'][index]['test'] = False
            data['nodes'][index]['val'] = False
        elif index in val:
            data['nodes'][index]['val'] = True
            data['nodes'][index]['test'] = False
        elif index in test:
            data['nodes'][index]['test'] = True
            data['nodes'][index]['val'] = False
        else:
            print('Errr', index)
        data['nodes'][index]['feature'] = row[1:].tolist()
        data['nodes'][index]['label'] = list(label.loc[label[label.columns[0]] == int(row[0])].values.flatten())[1:]
    edges_file = open(edges_file_name, 'r')
    edges = []
    for edge in edges_file:
        relation = {}
        tokens = edge.split(" ")
        source = int(tokens[0].strip())
        dest = int(tokens[1].strip())
        relation['source'] = source
        relation['target'] = dest
        if (search(source, val) or search(source, test)
            or search(dest, val) or search(dest, test)):
            relation['train_removed'] = True
        else:
            relation['train_removed'] = False
        edges.append(relation)
    data['links'] = edges
    with open('gds/gds-G.json', 'w') as outfile:
        json.dump(data, outfile)
'''
	Creates a class map of each node id to the corresponding class label
'''
def class_map(label_file_name):
    label = get_one_hot_labels(label_file_name)
    class_map = {}
    id_map = {}
    for index, row in label.iterrows():
        class_map[str(int(row[0]))] = list(row.values.flatten())[1:]
        id_map[str(int(row[0]))] = int(row[0])
    with open('gds/gds-class_map.json', 'w') as outfile:
        json.dump(class_map, outfile)
    with open('gds/gds-id_map.json', 'w') as outfile:
        json.dump(id_map, outfile)
def persist_embedding(embedding_file_name):
    embedding = get_embedding_df(embedding_file_name)
    embedding.drop(embedding.columns[0],axis=1, inplace=True)
    np.save('gds/gds-feats.npy', embedding)
#Generates the json input file with the labelled nodes which is splitted in training, test, validation sets
generate_G_Json('/home/local/SRI/e32049/dataset/gds/final/nodes_new_api/node_labels_final_reordered.txt',
               '/home/local/SRI/e32049/dataset/gds/final/nodes_new_api/node_features_length_reordered.txt',
                   '/home/local/SRI/e32049/dataset/gds/final/nodes_new_api/relationship_final_mysql_nodes_reordered.txt')
#Class map from actual labels to one hot encoded labels
class_map('/home/local/SRI/e32049/dataset/gds/final/nodes_new_api/node_labels_final_reordered.txt')
#Feature vectors for each node in the graph which can be learnt from unsupervised model like deep walk
# or chosen from the descriptor od cell like size, shape, orientation etc..
persist_embedding('/home/local/SRI/e32049/dataset/gds/final/nodes_new_api/node_features_length_reordered.txt', False)
