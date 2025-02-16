import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch_sparse import SparseTensor

def get_lightgcn_data(mode='cn'):
    if mode == 'cn':
        file_path = 'data/china/user-attraction/china.csv'
    else:
        file_path = 'data/australia/user-attraction/Australia.csv'
    df = pd.read_csv(file_path)
    user_mapping = {index: i for i, index in enumerate(df['user_ID'].unique())}
    attraction_mapping = {index: i for i, index in enumerate(df.columns[1:].unique())}
    edge_index = [[], []]
    for i in range(len(df)):
        for j in range(1, len(df.columns)):
            if df.iloc[i, j] == 1:
                edge_index[0].append(user_mapping[df['user_ID'][i]])
                edge_index[1].append(attraction_mapping[df.columns[j]])
    edge_index = torch.tensor(edge_index)

    # split the edges of the graph using a 60/10/30 train/validation/test split
    num_users, num_movies = len(user_mapping), len(attraction_mapping)
    num_interactions = edge_index.shape[1]
    all_indices = [i for i in range(num_interactions)]

    train_indices, test_indices = train_test_split(
        all_indices, test_size=0.3, random_state=1)
    # val_indices, test_indices = train_test_split(
    #     test_indices, test_size=0.25, random_state=1)

    train_edge_index = edge_index[:, train_indices]
    # val_edge_index = edge_index[:, val_indices]
    test_edge_index = edge_index[:, test_indices]

    # convert edge indices into Sparse Tensors: https://pytorch-geometric.readthedocs.io/en/latest/notes/sparse_tensor.html
    train_sparse_edge_index = SparseTensor(row=train_edge_index[0], col=train_edge_index[1], sparse_sizes=(
        num_users + num_movies, num_users + num_movies))
    # val_sparse_edge_index = SparseTensor(row=val_edge_index[0], col=val_edge_index[1], sparse_sizes=(
    #     num_users + num_movies, num_users + num_movies))
    test_sparse_edge_index = SparseTensor(row=test_edge_index[0], col=test_edge_index[1], sparse_sizes=(
        num_users + num_movies, num_users + num_movies))

    res = {}
    res['num_users'] = num_users
    res['num_movies'] = num_movies
    res['train_edge_index'] = train_edge_index
    # res['val_edge_index'] = val_edge_index
    res['test_edge_index'] = test_edge_index
    res['train_sparse_edge_index'] = train_sparse_edge_index
    # res['val_sparse_edge_index'] = val_sparse_edge_index
    res['test_sparse_edge_index'] = test_sparse_edge_index
    res['user_mapping'] = user_mapping
    res['attraction_mapping'] = attraction_mapping
    res['edge_index'] = edge_index
    return res