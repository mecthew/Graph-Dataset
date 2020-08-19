import os
import shutil
from ogb.nodeproppred import PygNodePropPredDataset
import scipy.sparse as sp
import numpy as np

from npz_loader import transform_to_autograph_format


def load_ogb(dataset_name, time_budget, zero_features=True, sample_num=None):
    print("*"*30, "Start!", "*"*30)
    dataset = PygNodePropPredDataset(name=dataset_name)

    split_idx = dataset.get_idx_split()
    train_idx, test_idx = split_idx["train"].numpy().tolist(), split_idx["valid"].numpy().tolist() + split_idx["test"].numpy().tolist()
    print("Train rate {}, test rate {}".format(len(train_idx)/(len(train_idx)+len(test_idx)), len(test_idx)/(len(train_idx)+len(test_idx))))

    graph = dataset[0]  # pyg graph object
    features, labels = graph.x.numpy(), graph.y.numpy()
    edge_index = graph.edge_index.numpy()
    edge_weight = graph.edge_attr
    print(features.shape, labels.shape, edge_index.shape)
    if zero_features:
        features = np.zeros((features.shape[0], features.shape[1]), dtype=np.float)
    if edge_weight is None:
        edge_weight = np.ones(edge_index.shape[1])

    adj_matrix = sp.coo_matrix((edge_weight.reshape(-1), edge_index), shape=(labels.shape[0], labels.shape[0]))
    node_indexs = np.arange(labels.shape[0])
    n_class = len(np.unique(labels))
    # output directory control
    output_dir = os.path.join(os.path.dirname(__file__) + '../data-offline')
    os.makedirs(output_dir, exist_ok=True)
    sample_num_str = "" if sample_num is None else str(sample_num)
    data_dir = os.path.join(output_dir, dataset_name+sample_num_str)
    if os.path.exists(data_dir):
        shutil.rmtree(data_dir)
    train_dir = os.path.join(data_dir, 'train.data')
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(train_dir, exist_ok=True)
    transform_to_autograph_format(features, labels, adj_matrix, node_indexs, n_class, data_dir, train_dir, time_budget)
    print("*"*30, "Finish!", "*"*30)


if __name__ == '__main__':
    load_ogb("ogbn-arxiv", zero_features=False, time_budget=500)
