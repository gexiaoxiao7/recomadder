from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.nn.conv import MessagePassing
from torch import nn, Tensor
from torch_sparse import SparseTensor, matmul
import torch
import numpy as np

class LightGCN(MessagePassing):
    """LightGCN Model as proposed in https://arxiv.org/abs/2002.02126
    """

    def __init__(self, num_users, num_items, embedding_dim=64, K=3, add_self_loops=False):
        """Initializes LightGCN Model

        Args:
            num_users (int): Number of users
            num_items (int): Number of items
            embedding_dim (int, optional): Dimensionality of embeddings. Defaults to 8.
            K (int, optional): Number of message passing layers. Defaults to 3.
            add_self_loops (bool, optional): Whether to add self loops for message passing. Defaults to False.
        """
        super().__init__()
        self.num_users, self.num_items = num_users, num_items
        self.embedding_dim, self.K = embedding_dim, K
        self.add_self_loops = add_self_loops

        self.users_emb = nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.embedding_dim) # e_u^0
        self.items_emb = nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.embedding_dim) # e_i^0

        nn.init.normal_(self.users_emb.weight, std=0.1)
        nn.init.normal_(self.items_emb.weight, std=0.1)

    def forward(self, edge_index: SparseTensor):
        """Forward propagation of LightGCN Model.

        Args:
            edge_index (SparseTensor): adjacency matrix

        Returns:
            tuple (Tensor): e_u_k, e_u_0, e_i_k, e_i_0
        """
        # compute \tilde{A}: symmetrically normalized adjacency matrix
        edge_index_norm = gcn_norm(
            edge_index, add_self_loops=self.add_self_loops)

        emb_0 = torch.cat([self.users_emb.weight, self.items_emb.weight]) # E^0
        embs = [emb_0]
        emb_k = emb_0

        # multi-scale diffusion
        for i in range(self.K):
            emb_k = self.propagate(edge_index_norm, x=emb_k)
            embs.append(emb_k)

        embs = torch.stack(embs, dim=1)
        emb_final = torch.mean(embs, dim=1) # E^K

        users_emb_final, items_emb_final = torch.split(
            emb_final, [self.num_users, self.num_items]) # splits into e_u^K and e_i^K

        # returns e_u^K, e_u^0, e_i^K, e_i^0
        return users_emb_final, self.users_emb.weight, items_emb_final, self.items_emb.weight

    def message(self, x_j: Tensor) -> Tensor:
        return x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        # computes \tilde{A} @ x
        return matmul(adj_t, x)


class AFM(nn.Module):
    def __init__(self, fields, embedding_dim, attention_dim, dropouts=0.25):
        super(AFM, self).__init__()
        self.num_fields = len(fields)
        self.embedding_dim = embedding_dim
        self.attention_dim = attention_dim
        self.offsets = np.array((0, *np.cumsum(fields)[:-1]), dtype=np.longlong)
        self.dropouts = dropouts

        # linner
        self.linear = torch.nn.Embedding(sum(fields) + 1, 1)
        self.bias = torch.nn.Parameter(torch.zeros((1,)))

        #embedding
        self.embedding = torch.nn.Embedding(sum(fields) + 1, embedding_dim)
        torch.nn.init.xavier_uniform_(self.embedding.weight.data)

        #attention
        self.attention = torch.nn.Linear(embedding_dim, attention_dim)
        self.projection = torch.nn.Linear(attention_dim, 1)
        self.fc = torch.nn.Linear(embedding_dim, 1)
        self.dropouts = dropouts
        self.output = nn.Linear(embedding_dim, 1)

    def forward(self, item_feature, user_embedding, item_embedding):
        # item_feature: (batch_size, num_fields)
        # user_embedding: (batch_size, embedding_dim)
        # item_embedding: (batch_size, embedding_dim)

        tmp = item_feature + item_feature.new_tensor(self.offsets).unsqueeze(0)  # (batch_size, num_fields)
        linear_part = torch.sum(self.linear(tmp), dim=1) + self.bias # (batch_size, 1)

        tmp = self.embedding(tmp) # (batch_size, num_fields, embedding_dim)
        # concat tmp with user_embedding and item_embedding
        user_embedding = user_embedding.unsqueeze(1) # (batch_size, 1, embedding_dim)
        item_embedding = item_embedding.unsqueeze(1) # (batch_size, 1, embedding_dim)
        tmp = torch.cat([tmp, item_embedding ,user_embedding], dim=1) # (batch_size, num_fields + 2, embedding_dim)

        num_fields = tmp.shape[1]
        row, col = [], []
        for i in range(num_fields - 1):
            for j in range(i+1, num_fields):
                row.append(i)
                col.append(j)
        p, q = tmp[:, row], tmp[:,col]
        inner = p * q
        attn_scores = nn.functional.relu(self.attention(inner))
        attn_scores = nn.functional.softmax(self.projection(attn_scores), dim=1)
        attn_scores = nn.functional.dropout(attn_scores, p = self.dropouts)
        attn_output = torch.sum(attn_scores * inner, dim = 1)
        attn_output = nn.functional.dropout(attn_output, p = self.dropouts)
        inner_attn_part = self.fc(attn_output)

        x = linear_part + inner_attn_part
        x = torch.sigmoid(x.squeeze(1))
        return x