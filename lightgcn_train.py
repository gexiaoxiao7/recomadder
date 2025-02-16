import argparse

import torch
from torch import nn, optim
import time
from data import get_lightgcn_data
from model import LightGCN
from utils import structured_negative_sampling, bpr_loss, get_metrics, sample_mini_batch
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, default='cn')
parser.add_argument('--k', type=int, default=5)
args = parser.parse_args()

res = get_lightgcn_data(args.mode)
user_mapping = res['user_mapping']
attraction_mapping = res['attraction_mapping']
edge_index = res['edge_index']
train_edge_index = res['train_edge_index']
train_sparse_edge_index = res['train_sparse_edge_index']
test_edge_index = res['test_edge_index']
test_sparse_edge_index = res['test_sparse_edge_index']
num_users, num_movies = len(user_mapping), len(attraction_mapping)



model = LightGCN(num_users, num_movies)




# define contants
ITERATIONS = 2000
BATCH_SIZE = 1024
LR = 1e-3
ITERS_PER_EVAL = 50
ITERS_PER_LR_DECAY = 200
K = args.k
LAMBDA = 1e-6

# setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
checkpoint_path = f'saved_models/{args.mode}/lightgcn_{K}/'
import os
if not os.path.exists(checkpoint_path):
    os.makedirs(checkpoint_path)

# device = torch.device('cpu')
print(f"Using device {device}.")

# model_dir = 'saved_models\lightgcn\lightgcn_0_loss-0.69113_test_ndcg0.03449.pth'
# model.load_state_dict(torch.load(model_dir))
model = model.to(device)
model.train()

optimizer = optim.Adam(model.parameters(), lr=LR)
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

edge_index = edge_index.to(device)
train_edge_index = train_edge_index.to(device)
train_sparse_edge_index = train_sparse_edge_index.to(device)
test_edge_index = test_edge_index.to(device)
test_sparse_edge_index = test_sparse_edge_index.to(device)

def evaluation(model, edge_index, sparse_edge_index, exclude_edge_indices, k, lambda_val):
    """Evaluates model loss and metrics including recall, precision, ndcg @ k

    Args:
        model (LighGCN): lightgcn model
        edge_index (torch.Tensor): 2 by N list of edges for split to evaluate
        sparse_edge_index (sparseTensor): sparse adjacency matrix for split to evaluate
        exclude_edge_indices ([type]): 2 by N list of edges for split to discount from evaluation
        k (int): determines the top k items to compute metrics on
        lambda_val (float): determines lambda for bpr loss

    Returns:
        tuple: bpr loss, recall @ k, precision @ k, ndcg @ k
    """
    # get embeddings
    users_emb_final, users_emb_0, items_emb_final, items_emb_0 = model.forward(
        sparse_edge_index)
    edges = structured_negative_sampling(
        edge_index, num_nodes=max(edge_index[1])+1 ,min_sample=len(edge_index[0]),contains_neg_self_loops=False)
    user_indices, pos_item_indices, neg_item_indices = edges[0], edges[1], edges[2]
    users_emb_final, users_emb_0 = users_emb_final[user_indices], users_emb_0[user_indices]
    pos_items_emb_final, pos_items_emb_0 = items_emb_final[
        pos_item_indices], items_emb_0[pos_item_indices]
    neg_items_emb_final, neg_items_emb_0 = items_emb_final[
        neg_item_indices], items_emb_0[neg_item_indices]

    loss = bpr_loss(users_emb_final, users_emb_0, pos_items_emb_final, pos_items_emb_0,
                    neg_items_emb_final, neg_items_emb_0, lambda_val).item()

    recall, precision, ndcg, hr = get_metrics(
        model, edge_index, exclude_edge_indices, k)

    return loss, recall, precision, ndcg, hr

# training loop

train_losses = []
val_losses = []

time_cur = time.time()
for iter in range(ITERATIONS):
    # forward propagation
    users_emb_final, users_emb_0, items_emb_final, items_emb_0 = model.forward(
        train_sparse_edge_index)
    # mini batching
    user_indices, pos_item_indices, neg_item_indices = sample_mini_batch(
        BATCH_SIZE, train_edge_index)

    user_indices, pos_item_indices, neg_item_indices = user_indices.to(
        device), pos_item_indices.to(device), neg_item_indices.to(device)
    users_emb_final, users_emb_0 = users_emb_final[user_indices], users_emb_0[user_indices]
    pos_items_emb_final, pos_items_emb_0 = items_emb_final[
        pos_item_indices], items_emb_0[pos_item_indices]
    neg_items_emb_final, neg_items_emb_0 = items_emb_final[
        neg_item_indices], items_emb_0[neg_item_indices]

    # loss computation
    train_loss = bpr_loss(users_emb_final, users_emb_0, pos_items_emb_final,
                          pos_items_emb_0, neg_items_emb_final, neg_items_emb_0, LAMBDA)


    optimizer.zero_grad()
    train_loss.backward()
    optimizer.step()

    if iter % ITERS_PER_EVAL == 0:
        model.eval()
        time_cur = time.time()
        val_loss, recall, precision, ndcg, hitrate = evaluation(
            model, test_edge_index, test_sparse_edge_index, [train_edge_index], K, LAMBDA)
        print(f"[Iteration {iter}/{ITERATIONS}] train_loss: {round(train_loss.item(), 5)}, val_loss: {round(val_loss, 5)}, val_recall@{K}: {round(recall, 5)}, val_precision@{K}: {round(precision, 5)}, val_ndcg@{K}: {round(ndcg, 5)}"
              f", val_hitrate@{K}: {round(hitrate, 5)}")
        # save model
        torch.save(model.state_dict(), f"{checkpoint_path}lightgcn_{iter}_recall{round(recall, 5)}_precision{round(precision, 5)}_ndcg@{K}_{round(ndcg, 5)}_hr{round(hitrate, 5)}.pth")
        train_losses.append(train_loss.item())
        val_losses.append(val_loss)
        model.train()

        iters = [iter * ITERS_PER_EVAL for iter in range(len(train_losses))]
        plt.plot(iters, train_losses, label='train')
        plt.plot(iters, val_losses, label='validation')
        plt.xlabel('iteration')
        plt.ylabel('loss')
        plt.title('training and validation loss curves')
        plt.legend()
        # save plot
        if not os.path.exists(f'outputs/{args.mode}_{args.k}'):
            os.makedirs(f'outputs/{args.mode}_{args.k}')
        plt.savefig(f'outputs/{args.mode}_{args.k}/loss_curve{time.time()}.png')
        plt.clf()

    if iter % ITERS_PER_LR_DECAY == 0 and iter != 0:
        scheduler.step()
