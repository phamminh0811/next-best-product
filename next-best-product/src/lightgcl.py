import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import sparse_dropout


class LightGCL(nn.Module):
    """LightGCL (Cai et al., ICLR 2023), adapted from the official HKUDS/LightGCL code.

    The training loss is unchanged. Differences from the original: the test phase runs on any
    device and scores with the current parameters (predict / embeddings) instead of the embeddings
    cached by the last training step, and the fixed graph tensors are buffers so .to(device) moves them.
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

    def propagate(self, dropout):
        """Layer-summed embeddings of the graph view (E_u, E_i) and of the SVD view (G_u, G_i)."""
        E_u_list, E_i_list = [self.E_u_0], [self.E_i_0]
        G_u_list, G_i_list = [self.E_u_0], [self.E_i_0]
        for layer in range(1, self.l + 1):
            # GNN propagation
            Z_u = torch.spmm(sparse_dropout(self.adj_norm, dropout), E_i_list[layer - 1])
            Z_i = torch.spmm(sparse_dropout(self.adj_norm, dropout).transpose(0, 1), E_u_list[layer - 1])

            # svd_adj propagation
            vt_ei = self.vt @ E_i_list[layer - 1]
            G_u_list.append(self.u_mul_s @ vt_ei)
            ut_eu = self.ut @ E_u_list[layer - 1]
            G_i_list.append(self.v_mul_s @ ut_eu)

            # aggregate
            E_u_list.append(Z_u)
            E_i_list.append(Z_i)

        # aggregate across layers
        return sum(E_u_list), sum(E_i_list), sum(G_u_list), sum(G_i_list)

    @torch.no_grad()
    def embeddings(self):
        """Final user and item embeddings (graph view) from the current parameters, without edge dropout."""
        E_u, E_i, _, _ = self.propagate(dropout=0.0)
        return E_u, E_i

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
        neg_score = torch.log(torch.exp(G_u[uids] @ E_u.T / self.temp).sum(1) + 1e-8).mean()
        neg_score += torch.log(torch.exp(G_i[iids] @ E_i.T / self.temp).sum(1) + 1e-8).mean()
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
