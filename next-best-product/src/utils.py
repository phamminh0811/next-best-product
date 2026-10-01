import numpy as np
import scipy.sparse as sps
import torch
import torch.nn as nn

def sparse_dropout(mat, dropout):
    if dropout == 0.0:
        return mat
    indices = mat.indices()
    values = nn.functional.dropout(mat.values(), p=dropout)
    size = mat.size()
    # Same indices as mat, which indices() only returns for a valid coalesced tensor.
    return torch.sparse_coo_tensor(indices, values, size, check_invariants=False, is_coalesced=True)

def spmm(sp, emb, device):
    sp = sp.coalesce()
    cols = sp.indices()[1]
    rows = sp.indices()[0]
    col_segs =  emb[cols] * torch.unsqueeze(sp.values(),dim=1)
    result = torch.zeros((sp.shape[0],emb.shape[1]), device=device)
    result.index_add_(0, rows, col_segs)
    return result

# Graph construction, training loop and metrics of the official main.py / utils.py, shared by the notebooks.

def normalize_adj(mat):
    """D_u^{-1/2} R D_i^{-1/2} of a users x items scipy matrix, as a float32 coo matrix (official main.py)."""
    coo = mat.tocoo()
    row_deg = np.asarray(mat.sum(1)).ravel()
    col_deg = np.asarray(mat.sum(0)).ravel()
    values = coo.data / np.sqrt(row_deg[coo.row] * col_deg[coo.col])
    return sps.coo_matrix((values.astype(np.float32), (coo.row, coo.col)), shape=coo.shape)

def to_torch_sparse(coo, device):
    indices = torch.from_numpy(np.vstack([coo.row, coo.col]).astype(np.int64))
    values = torch.from_numpy(coo.data.astype(np.float32))
    return torch.sparse_coo_tensor(indices, values, coo.shape, check_invariants=True).coalesce().to(device)

def svd_view(adj_norm, q, exact=False):
    """Rank-q SVD factors in the order LightGCL takes them: u_mul_s, v_mul_s, ut, vt.

    exact=False uses torch.svd_lowrank as the official code does; it is randomized and underestimates the
    smaller of the q singular values. exact=True keeps the top q of a full SVD of the dense matrix, which is
    affordable when one side has few nodes (Expedia: 665k users x 100 clusters).
    """
    if exact:
        U, s, Vh = torch.linalg.svd(adj_norm.to_dense(), full_matrices=False)
        svd_u, s, svd_v = U[:, :q], s[:q], Vh[:q].T
    else:
        svd_u, s, svd_v = torch.svd_lowrank(adj_norm, q=q)
    return svd_u @ torch.diag(s), svd_v @ torch.diag(s), svd_u.T, svd_v.T

def sample_negatives(edge_users, edge_items, n_items, generator):
    """One item per edge, uniform over the items its user has no edge to (TrnData.neg_sampling, vectorized)."""
    edge_keys = edge_users * n_items + edge_items
    negs = torch.randint(n_items, edge_users.shape, device=edge_users.device, generator=generator)
    for _ in range(100):
        clash = torch.isin(edge_users * n_items + negs, edge_keys)
        if not clash.any():
            break
        negs[clash] = torch.randint(n_items, (int(clash.sum()),), device=edge_users.device, generator=generator)
    return negs

def train_epoch(model, optimizer, edge_users, edge_items, n_items, batch_size, generator, progress=None):
    """One pass over all edges in random order; returns mean (loss, loss_r, loss_s).

    A fresh negative is sampled per edge when model.uses_negatives, else neg is None and iids holds the positives only.
    progress, if given, wraps the batch range (e.g. functools.partial(tqdm, leave=False)).
    """
    negs = sample_negatives(edge_users, edge_items, n_items, generator) if model.uses_negatives else None
    order = torch.randperm(len(edge_users), device=edge_users.device, generator=generator)
    starts = range(0, len(order), batch_size)
    total = torch.zeros(3, device=edge_users.device)
    for start in (progress(starts) if progress else starts):
        batch = order[start:start + batch_size]
        uids, pos = edge_users[batch], edge_items[batch]
        neg = negs[batch] if negs is not None else None
        iids = torch.cat([pos, neg]) if neg is not None else pos
        loss, loss_r, loss_s = model(uids, iids, pos, neg)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += torch.stack([loss, loss_r, loss_s]).detach()
    return (total / len(starts)).tolist()

def topk_metrics(scores, labels, topks):
    """Per-user Recall@K and NDCG@K of the official metrics(): recall = hits / |labels|, NDCG normalized by
    the ideal DCG of min(K, |labels|) items. scores (B x I) float, labels (B x I) bool; users without labels get NaN."""
    k_max = max(topks)
    discount = 1.0 / torch.log2(torch.arange(2, k_max + 2, device=scores.device, dtype=torch.float32))
    idcg = torch.cat([torch.zeros(1, device=scores.device), discount.cumsum(0)])
    hits = labels.gather(1, scores.topk(k_max, dim=1).indices).float()
    n_labels = labels.sum(1)
    out = {"n_labels": n_labels}
    for k in topks:
        out[f"recall@{k}"] = hits[:, :k].sum(1) / n_labels
        out[f"ndcg@{k}"] = (hits[:, :k] * discount[:k]).sum(1) / idcg[n_labels.clamp(max=k)]
    return out