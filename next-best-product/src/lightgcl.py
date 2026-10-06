import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import sparse_dropout


def log_sum_exp(scores):
    """log(sum(exp(scores), dim=1)), shifted by the row max so it cannot overflow. Overwrites scores in place, so the
    batch x nodes matrix exists once (torch.logsumexp keeps about one more copy: +2.5 GiB on Expedia)."""
    row_max = scores.max(dim=1, keepdim=True).values.detach()
    return row_max.squeeze(1) + scores.sub_(row_max).exp_().sum(1).log()


class LightGCL(nn.Module):
    """LightGCL (Cai et al., ICLR 2023), adapted from the official HKUDS/LightGCL code.

    The training loss is unchanged. Differences from the original: the test phase runs on any
    device and scores with the current parameters (predict / embeddings) instead of the embeddings
    cached by the last training step, the fixed graph tensors are buffers so .to(device) moves them, and
    the contrastive denominators use a max-shifted log_sum_exp instead of log(sum(exp(.)) + 1e-8), which
    overflows to inf once a score / temp passes ~88 (seen with the denoised view of radar.py on Expedia).
    """

    def __init__(self, n_u, n_i, d, u_mul_s, v_mul_s, ut, vt, train_csr, adj_norm, l, temp, lambda_1, lambda_2, dropout):
        super(LightGCL, self).__init__()
        self.E_u_0 = nn.Parameter(nn.init.xavier_uniform_(torch.empty(n_u, d)))
        self.E_i_0 = nn.Parameter(nn.init.xavier_uniform_(torch.empty(n_i, d)))

        # Normalized adjacency and its SVD factors are inputs, not weights: non-persistent buffers
        # follow model.to(device) but stay out of the state_dict.
        self.register_buffer("adj_norm", adj_norm, persistent=False)
        self.register_buffer("u_mul_s", u_mul_s, persistent=False)
        self.register_buffer("v_mul_s", v_mul_s, persistent=False)
        self.register_buffer("ut", ut, persistent=False)
        self.register_buffer("vt", vt, persistent=False)
        self.train_csr = train_csr  # scipy csr, users x items; predict() masks these items

        self.l = l
        self.temp = temp
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.dropout = dropout

    def graph_layers(self, dropout):
        """Per-layer embeddings of the graph view: lists of l + 1 user and item tensors, layer 0 first."""
        E_u_list, E_i_list = [self.E_u_0], [self.E_i_0]
        for layer in range(1, self.l + 1):
            # GNN propagation
            Z_u = torch.spmm(sparse_dropout(self.adj_norm, dropout), E_i_list[layer - 1])
            Z_i = torch.spmm(sparse_dropout(self.adj_norm, dropout).transpose(0, 1), E_u_list[layer - 1])
            E_u_list.append(Z_u)
            E_i_list.append(Z_i)
        return E_u_list, E_i_list

    def view(self, E_u_list, E_i_list):
        """Layer-summed embeddings of the SVD view: layer l propagates layer l - 1 of the graph view over U S V^T."""
        G_u_list, G_i_list = [E_u_list[0]], [E_i_list[0]]
        for E_u, E_i in zip(E_u_list[:-1], E_i_list[:-1]):
            G_u_list.append(self.u_mul_s @ (self.vt @ E_i))
            G_i_list.append(self.v_mul_s @ (self.ut @ E_u))
        return sum(G_u_list), sum(G_i_list)

    def propagate(self, dropout):
        """Layer-summed embeddings of the graph view (E_u, E_i) and of the contrastive view (G_u, G_i)."""
        E_u_list, E_i_list = self.graph_layers(dropout)
        G_u, G_i = self.view(E_u_list, E_i_list)
        return sum(E_u_list), sum(E_i_list), G_u, G_i

    @torch.no_grad()
    def embeddings(self):
        """Final user and item embeddings (graph view) from the current parameters, without edge dropout."""
        E_u_list, E_i_list = self.graph_layers(dropout=0.0)
        return sum(E_u_list), sum(E_i_list)

    @torch.no_grad()
    def predict(self, uids, exclude_seen=True):
        """Scores (len(uids) x n_items) for graph user indices uids; with exclude_seen, items the user
        has in train_csr score -inf. Propagates the whole graph on every call."""
        E_u, E_i = self.embeddings()
        scores = E_u[uids] @ E_i.T
        if exclude_seen:
            seen = torch.as_tensor(self.train_csr[uids.cpu().numpy()].toarray() > 0, device=scores.device)
            scores = scores.masked_fill(seen, -torch.inf)
        return scores

    uses_negatives = True  # training feeds one sampled negative item per positive

    def contrastive_loss(self, E_u, E_i, G_u, G_i, uids, iids):
        neg_score = log_sum_exp(G_u[uids] @ E_u.T / self.temp).mean()
        neg_score += log_sum_exp(G_i[iids] @ E_i.T / self.temp).mean()
        pos_score = (torch.clamp((G_u[uids] * E_u[uids]).sum(1) / self.temp, -5.0, 5.0)).mean() + (torch.clamp((G_i[iids] * E_i[iids]).sum(1) / self.temp, -5.0, 5.0)).mean()
        return -pos_score + neg_score

    def recommendation_loss(self, E_u, E_i, uids, pos, neg):
        """BPR with one sampled negative per positive."""
        u_emb = E_u[uids]
        pos_emb = E_i[pos]
        neg_emb = E_i[neg]
        pos_scores = (u_emb * pos_emb).sum(-1)
        neg_scores = (u_emb * neg_emb).sum(-1)
        return -(pos_scores - neg_scores).sigmoid().log().mean()

    def reg_loss(self):
        loss_reg = 0
        for param in self.parameters():
            loss_reg += param.norm(2).square()
        return loss_reg * self.lambda_2

    def forward(self, uids, iids, pos, neg, test=False):
        if test:  # testing phase: every item ranked best first, train items last
            return self.predict(uids).argsort(descending=True)

        # training phase
        E_u, E_i, G_u, G_i = self.propagate(self.dropout)
        loss_s = self.contrastive_loss(E_u, E_i, G_u, G_i, uids, iids)
        loss_r = self.recommendation_loss(E_u, E_i, uids, pos, neg)
        loss = loss_r + self.lambda_1 * loss_s + self.reg_loss()
        return loss, loss_r, self.lambda_1 * loss_s


class LightGCLSoftmax(LightGCL):
    """LightGCL with a full-softmax recommendation loss instead of BPR with sampled negatives.

    Meant for small catalogs such as Expedia's 100 hotel clusters: each positive is scored against every
    item, so no negatives are sampled, and every item is an anchor of the item-side contrastive loss
    (BPR training uses the batch's positives and negatives there). Graph, SVD view and user-side
    contrastive loss are unchanged.
    """

    uses_negatives = False

    def contrastive_loss(self, E_u, E_i, G_u, G_i, uids, iids):
        all_items = torch.arange(E_i.shape[0], device=E_i.device)
        return super().contrastive_loss(E_u, E_i, G_u, G_i, uids, all_items)

    def recommendation_loss(self, E_u, E_i, uids, pos, neg):
        """Cross-entropy of each positive item against all items (neg is unused)."""
        return F.cross_entropy(E_u[uids] @ E_i.T, pos)


class LightGCLTimeSoftmax(LightGCLSoftmax):
    """LightGCLSoftmax whose graph view aggregates with the time-aware weights of GraphPro (Yang et al., WWW 2024).

    A neighbor v of node u weighs adj_norm[u, v] / 2 + alpha[u, v] / 2, where alpha is the softmax over u's neighbors
    of their interaction times scaled to [0, 1], so recent neighbors count more (GraphPro, Eq. 5–6, with adj_norm in
    place of LightGCN's 1 / sqrt(|N_u| |N_v|)). time_u (users x items) holds alpha over each user's items and time_i
    (items x users) over each item's users, both with the sparsity of adj_norm. The SVD view and the contrastive loss
    still use adj_norm alone.
    """

    def __init__(self, *args, time_u, time_i, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_buffer("adj_u", (0.5 * self.adj_norm + 0.5 * time_u).coalesce(), persistent=False)
        self.register_buffer("adj_i", (0.5 * self.adj_norm.transpose(0, 1) + 0.5 * time_i).coalesce(), persistent=False)

    def graph_layers(self, dropout):
        E_u_list, E_i_list = [self.E_u_0], [self.E_i_0]
        for layer in range(1, self.l + 1):
            E_u_list.append(torch.spmm(sparse_dropout(self.adj_u, dropout), E_i_list[layer - 1]))
            E_i_list.append(torch.spmm(sparse_dropout(self.adj_i, dropout), E_u_list[layer - 1]))
        return E_u_list, E_i_list
