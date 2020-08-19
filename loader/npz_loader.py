import numpy as np
import scipy.sparse as sp
import pandas as pd
import os
import yaml
import shutil


def drop_n_unique(x, n=1):
    """
    :param x: pd.DataFrame.
    :param n: int, threshold of unique.
    :return:
    """
    drop_cols = []
    for col in x:
        if x[col].nunique() == n:
            drop_cols.append(col)
    # print(f"Drop {drop_cols} by condition (nunique={n})")
    all_zero = len(drop_cols) == len(x.columns)
    x.drop(columns=drop_cols, inplace=True, axis=1)
    print(f"Remain cols {len(x.columns)}")
    return all_zero


def load_npz(dataset, npz_dir='../npz-data'):
    file_map = {'az-cs': 'amazon_electronics_computers.npz', 'az-po': 'amazon_electronics_photo.npz',
                'co-cs': 'ms_academic_cs.npz', 'co-phy': 'ms_academic_phy.npz'}
    if dataset in file_map:
        file_name = file_map[dataset]
    else:
        file_name = dataset + ".npz"
    file_path = os.path.join(npz_dir, file_name)
    with np.load(file_path, allow_pickle=True) as loader:
        loader = dict(loader)
        adj_matrix = sp.csr_matrix((loader['adj_data'], loader['adj_indices'], loader['adj_indptr']),
                                   shape=loader['adj_shape'])

        if 'attr_data' in loader:
            # Attributes are stored as a sparse CSR matrix
            attr_matrix = sp.csr_matrix((loader['attr_data'], loader['attr_indices'], loader['attr_indptr']),
                                        shape=loader['attr_shape'])
        elif 'attr_matrix' in loader:
            # Attributes are stored as a (dense) np.ndarray
            attr_matrix = loader['attr_matrix']
        else:
            attr_matrix = None

        if 'labels_data' in loader:
            # Labels are stored as a CSR matrix
            labels = sp.csr_matrix((loader['labels_data'], loader['labels_indices'], loader['labels_indptr']),
                                   shape=loader['labels_shape'])
        elif 'labels' in loader:
            # Labels are stored as a numpy array
            labels = loader['labels']
        else:
            labels = None

    if sp.isspmatrix(attr_matrix):
        attr_matrix = attr_matrix.todense()
    if sp.isspmatrix(labels):
        labels = labels.todense()
    adj_matrix = adj_matrix.tocoo()

    return attr_matrix, labels, adj_matrix


def transform_to_autograph_format(features, labels, adj, node_indexs, n_class, output_dir, train_dir, time_budget=150):
    if len(labels.shape) > 1 and labels.shape[1] > 1:
        labels = np.argmax(labels, axis=1)
    elif len(labels.shape) > 1 and labels.shape[1] == 1:
        labels = labels.reshape(-1)
    train_indices, test_indices = [], []
    for i in range(n_class):
        indices_i = np.where(labels == i)[0]
        train_ind = np.random.choice(indices_i, int(len(indices_i) * 0.4)+1, replace=False).tolist()
        test_ind = list(set(indices_i) - set(train_ind))
        train_indices += train_ind
        test_indices += test_ind
    train_indices = np.sort(train_indices)
    test_indices = np.sort(test_indices)
    print("train samples {}, test samples {}".format(len(train_indices), len(test_indices)))

    # train_indices = np.sort(np.random.choice(node_indexs, int(len(node_indexs) * 0.4), replace=False)).tolist()
    # test_indices = np.sort(list(set(node_indexs) - set(train_indices))).tolist()
    feat_table = pd.DataFrame(features)
    feat_cols = ["f{}".format(i) for i in range(features.shape[1])]
    feat_table.columns = feat_cols
    feat_table.insert(0, "node_index", node_indexs)
    feat_table.to_csv(os.path.join(train_dir, 'feature.tsv'), index=False, sep='\t')
    print("finish feature.tsv, features shape {}".format(features.shape))

    train_labels = labels[train_indices]
    train_label_tsv = pd.DataFrame({"node_index": train_indices, "label": train_labels.tolist()})
    train_label_tsv.to_csv(os.path.join(train_dir, "train_label.tsv"), index=False, sep='\t')
    print("finish train_label.tsv")

    test_labels = labels[test_indices]
    test_label_tsv = pd.DataFrame({"node_index": test_indices, "label": test_labels.tolist()})
    test_label_tsv.to_csv(os.path.join(output_dir, "test_label.tsv"), index=False, sep='\t')
    print("finish test_label.tsv")

    edge_tsv = pd.DataFrame({"src_idx": adj.row, "dst_idx": adj.col, "edge_weight": adj.data})
    edge_tsv.to_csv(os.path.join(train_dir, "edge.tsv"), index=False, sep='\t')
    print("finish edge.tsv")

    with open(os.path.join(train_dir, "train_node_id.txt"), 'w', encoding='utf8') as fout:
        for idx in train_indices:
            fout.write(str(idx) + '\n')
    with open(os.path.join(train_dir, "test_node_id.txt"), 'w', encoding='utf8') as fout:
        for idx in test_indices:
            fout.write(str(idx) + '\n')

    schema_dict = {feat_name: "num" for feat_name in feat_cols}
    yml_dict = {'n_class': n_class,
                'schema': schema_dict,
                'time_budget': time_budget}
    with open(os.path.join(train_dir, "config.yml"), 'w', encoding='utf8') as f:
        yaml.dump(yml_dict, f)


def npz_to_autograph(dataset, remove_selfloop, npz_dir='../npz-data', sample_num=None, time_budget=150):
    print("*"*30, "Start!", "*"*30)
    features, labels, adj_matrix = load_npz(dataset, npz_dir=npz_dir)
    assert features.shape[0] == labels.shape[0]
    raw_data_info = {
        "n_class": len(np.unique(labels)),
        "num_node": labels.shape[0],
        "num_edge": len(adj_matrix.data),
        "undirected_edge_num": len(adj_matrix.data) / 2,
        "num_attr": features.shape[1],
        "features_only_one_and_zero": not ((features != 0) & (features != 1)).any(),
        "edge_weight_only_ones": (adj_matrix.data == 1).all()
    }
    print("{}: raw data info \n{}".format(dataset, raw_data_info))
    print(len(np.unique(np.where((features != 0.0) & (features != 1.0))[0])))

    adj_matrix = adj_matrix.tocsr()
    adj_matrix = adj_matrix + adj_matrix.T
    adj_matrix[adj_matrix != 0] = 1.0
    if remove_selfloop:
        adj_matrix = adj_matrix.tolil()
        for i in range(labels.shape[0]):
            adj_matrix[i, i] = 0.0
    adj_matrix = adj_matrix.tocoo()

    if sample_num:
        random_indices = np.sort(np.random.choice(np.arange(labels.shape[0]), sample_num, replace=False))
        old2new = {random_indices[idx]: idx for idx in range(len(random_indices))}
        labels = labels[random_indices]
        features = features[random_indices]
        new_row, new_col, new_data = [], [], []
        for idx, (i, j) in enumerate(zip(adj_matrix.row, adj_matrix.col)):
            if i in random_indices and j in random_indices:
                new_row.append(old2new[i])
                new_col.append(old2new[j])
                new_data.append(adj_matrix.data[idx])
        adj_matrix = sp.coo_matrix((new_data, (new_row, new_col)), shape=(labels.shape[0], labels.shape[0]))

    n_class = len(np.unique(labels))
    node_indexs = np.arange(labels.shape[0])
    processed_data_info = {
        "n_class": len(np.unique(labels)),
        "num_node": labels.shape[0],
        "num_edge": len(adj_matrix.data),
        "undirected_edge_num": len(adj_matrix.data) / 2,
        "num_attr": features.shape[1],
        "features_only_one_and_zero": not ((features != 0) & (features != 1)).any(),
        "edge_weight_only_ones": (adj_matrix.data == 1).all()
    }
    print("{}: processed data info: \n{}".format(dataset, processed_data_info))

    # output directory control
    output_dir = os.path.dirname(__file__) + '/../data-offline'
    os.makedirs(output_dir, exist_ok=True)
    sample_num_str = "" if sample_num is None else str(sample_num)
    data_dir = os.path.join(output_dir, dataset+sample_num_str)
    if os.path.exists(data_dir):
        shutil.rmtree(data_dir)
    train_dir = os.path.join(data_dir, 'train.data')
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(train_dir, exist_ok=True)
    transform_to_autograph_format(features, labels, adj_matrix, node_indexs, n_class, data_dir, train_dir, time_budget)
    print("*"*30, "Finish!", "*"*30)


if __name__ == '__main__':
    # npz_to_autograph('az-cs', remove_selfloop=True, sample_num=None, time_budget=200)
    # npz_to_autograph('az-po', remove_selfloop=True, sample_num=None, time_budget=200)
    npz_to_autograph('cora_full', remove_selfloop=True, sample_num=None, time_budget=200)
    # npz_to_autograph('co-phy', remove_selfloop=True, sample_num=None, time_budget=300)
