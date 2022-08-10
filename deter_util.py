r"""
Deterministic implementation for tensor operations
"""
import torch


def det_softmax(self, src, index, num_nodes, dim=0):
    """
    Perform sparse softmax
    References:
    * https://pytorch-geometric.readthedocs.io/en/latest/modules/utils.html#torch_geometric.utils.softmax
    """
    from torch_scatter import scatter_max
    N = num_nodes
    src_max, _ = scatter_max(src, index, dim, dim_size=N)
    src_max = src_max.index_select(dim, index)
    out = (src - src_max).exp()
    out_sum = torch.zeros(out.shape, device=src.device)
    out_sum = self.deter_scatter_add_(index, out, out_sum)
    out_sum = out_sum.index_select(dim, index)
    return out / (out_sum + 1e-16)


def deter_scatter_add_(index, src_emb, out):
    import torch
    from torch_scatter import segment_csr, scatter_max
    # currently 1 dim only
    assert len(index.shape) == 1
    in_dim = src_emb.shape[0]
    out_dim = out.shape[0]
    device = index.device
    index_reorder, indices = torch.sort(index)
    max_from = torch.full([out_dim], fill_value=-1, dtype=torch.long, device=device)
    scatter_max(src=torch.arange(0, in_dim, dtype=torch.long, device=device),
                index=index_reorder, dim=0, out=max_from)
    max_from += 1
    max_from, _ = torch.cummax(max_from, dim=0)
    ind_ptr = torch.cat([torch.tensor([0], dtype=torch.long, device=device), max_from], dim=0)
    src_reorder = src_emb[indices]
    return segment_csr(src_reorder, ind_ptr, out)
