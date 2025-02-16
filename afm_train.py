import torch
import pandas as pd
from sklearn.model_selection import train_test_split

from data import get_lightgcn_data
from model import LightGCN, AFM
import argparse
import numpy as np
from sklearn.preprocessing import LabelEncoder
from torch import nn, optim

from utils import sample_mini_batch

parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, default='au')
parser.add_argument('--k', type=int, default=5)
parser.add_argument('--lmodel', type=str, default='saved_models/au/lightgcn_10/lightgcn_200_recall0.06004_precision0.08459_ndcg@100.09466_hr0.31964.pth')
args = parser.parse_args()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
K = args.k
checkpoint_path = f'saved_models/{args.mode}/afm_{K}/'
import os
if not os.path.exists(checkpoint_path):
    os.makedirs(checkpoint_path)
def load_data(cfg):
    res = get_lightgcn_data(cfg.mode)



    if cfg.mode == 'cn':
        features = pd.read_csv('data/china/features.csv',encoding='utf-8')
    else:
        features = pd.read_csv('data/australia/features.csv', encoding='utf-8')

    user_mapping = res['user_mapping']
    attraction_mapping = res['attraction_mapping']

    features['attractionID'] = features['attractionID'].apply(lambda x: attraction_mapping[x])
    item_features = torch.full([ len(attraction_mapping) ,2], 0)

    data_X = features[['emotion', 'tag']]
    data_X = data_X.apply(LabelEncoder().fit_transform)
    fields = []
    for col in data_X.columns:
        tmp = data_X[col].max() + 1
        fields.append(tmp)
    fields = np.array(fields)

    for i in range(len(features)):
        item_features[features['attractionID'][i]] = torch.tensor([data_X[data_X.columns[0]][i], data_X[data_X.columns[1]][i]])

    num_users = res['num_users']
    num_attractions = res['num_movies']
    lightgcn_model = LightGCN(num_users, num_attractions)
    model_dir = cfg.lmodel
    lightgcn_model.load_state_dict(torch.load(model_dir))
    lightgcn_model.to(device)
    lightgcn_model.eval()
    E_user_final = lightgcn_model.users_emb.weight
    E_item_final = lightgcn_model.items_emb.weight

    interactions = torch.zeros(num_users, num_attractions)
    for i in range(res['edge_index'].shape[0]):
        interactions[res['edge_index'][0][i]][res['edge_index'][1][i]] = 1

    return (item_features.to(device), # (num_attractions, feature_dim)
            E_user_final.clone().detach().to(device), # (num_users, embedding_dim)
            E_item_final.clone().detach().to(device), res, fields)# (num_attractions, embedding_dim)

def evaluation(model, edge_index, exclude_edge_indices, k, lambda_val = 1e-6):
    pass

def train_afm(model, res, fields ,item_features, E_user_final, E_item_final, epochs=10, batch_size=512, lr=0.001):

    edge_index = res['edge_index'].to(device)
    train_edge_index = res['train_edge_index'].to(device)
    train_sparse_edge_index = res['train_sparse_edge_index'].to(device)
    test_edge_index = res['test_edge_index'].to(device)
    test_sparse_edge_index = res['test_sparse_edge_index'].to(device)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    train_losses = []
    val_losses = []
    for epoch in range(epochs):
        model.train()
        # mini batching
        user_indices, pos_item_indices, neg_item_indices = sample_mini_batch(batch_size, train_edge_index)
        user_indices, pos_item_indices, neg_item_indices = user_indices.to(
            device), pos_item_indices.to(device), neg_item_indices.to(device)
        user_ids = torch.cat([user_indices, user_indices])
        item_ids = torch.cat([pos_item_indices, neg_item_indices])
        labels = torch.cat([torch.ones(len(user_indices)), torch.zeros(len(user_indices))])
        labels = labels.to(device)
        item_feats = item_features[item_ids]
        user_embs = E_user_final[user_ids]
        item_embs = E_item_final[item_ids]
        outputs = model(item_feats, user_embs, item_embs)
        loss = criterion(outputs, labels.float().detach())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % 1 == 0:
            model.eval()

            # print(
            #     f"[Iteration {iter}/{epochs}] train_loss: {round(loss.item(), 5)}, val_loss: {round(val_loss, 5)}, val_recall@{K}: {round(recall, 5)}, val_precision@{K}: {round(precision, 5)}, val_ndcg@{K}: {round(ndcg, 5)}"
            #     f", val_hitrate@{K}: {round(hitrate, 5)}")
            # # save model
            # torch.save(model.state_dict(),
            #            f"{checkpoint_path}{iter}_recall{round(recall, 5)}_precision{round(precision, 5)}_ndcg@{K}_{round(ndcg, 5)}_hr{round(hitrate, 5)}.pth")
            # train_losses.append(loss.item())
            # val_losses.append(val_loss)
            # model.train()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.4f}")



# 生成候选列表
# candidate_lists = {}
# for user in range(num_users):
#     e_u = lightgcn_model.users_emb.weight[user]
#     scores = lightgcn_model.items_emb.weight @ e_u
#     values, indices = torch.topk(scores, k=100)
#     candidate_lists[user] = indices

if __name__ == '__main__':
    print("============================LOADING DATA...======================================")
    item_features, E_user_final, E_item_final,res, fields = load_data(args)
    feature_dim = item_features.shape[1]
    embedding_dim = E_user_final.shape[1]
    attention_dim = 16
    afm_model = AFM(fields, embedding_dim, attention_dim)
    afm_model.to(device)
    print("============================START TRAINING========================================")
    train_afm(afm_model,res,fields,item_features, E_user_final, E_item_final)