import torch
import torch_geometric


class Graph(torch_geometric.data.Data):
    num_nodes: int
    num_edges: int
    edge_index: torch.LongTensor  # torch.Size([2, E]), the directed edges

    r"""
    -1 means that it's masked
    """
    x: torch.LongTensor  # torch.Size([N]), the original ids of the entities
    edge_attr: torch.LongTensor  # torch.Size([E]), the relation types of the edges

    def __init__(self, *args, **kwargs):
        super(Graph, self).__init__(*args, **kwargs)


class GraphWithAnswer(Graph):
    x_query: torch.LongTensor  # torch.Size([Q]), a list of nodes to be predicted
    x_ans: torch.LongTensor  # torch.Size([Q]), the answer to x_query
    edge_query: torch.LongTensor  # torch.Size([Q]), a list of edges to be predicted. If not available, use edge_attr == -1.
    edge_ans: torch.LongTensor  # torch.Size([Q]), the answer to edge_attr == -1
    x_pred_mask: torch.Tensor  # torch.Size([Q, F]), the True positions should be ignored in loss
    x_pred_weight: torch.Tensor  # torch.Size([Q]), the weight for the query in the loss
    joint_nodes: torch.Tensor  # torch.Size([2Q]), the nodes to be united
    union_query: torch.Tensor  # torch.Size([Q]), the query node of union operation

    def __init__(self, *args, **kwargs):
        super(GraphWithAnswer, self).__init__(*args, **kwargs)

    def __inc__(self, key, value, *args, **kwargs):
        if key == 'x_query':
            return self.num_nodes
        elif key == 'edge_query':
            return self.num_edges
        elif key == 'joint_nodes':
            return self.num_nodes
        elif key == 'union_query':
            return self.num_nodes
        elif key == 'x_pred_mask' and self.x_pred_mask.dtype == torch.long:
            return torch.tensor([[len(self.x_query)], [0]], device=value.device)
        else:
            return super(GraphWithAnswer, self).__inc__(key, value, *args, **kwargs)

    def __cat_dim__(self, key, value, *args, **kwargs):
        if key == 'x_pred_mask' and self.x_pred_mask.dtype == torch.long:
            return 1
        else:
            return super(GraphWithAnswer, self).__cat_dim__(key, value, *args, **kwargs) or 0

    def get_x_pred_indices(self):
        r"""
        :return: an object that is suitable for torch tensor index selection
        from the list of all nodes to Q items
        """
        if hasattr(self, 'x_query') and self.x_query is not None:
            return self.x_query
        return (self.x == -1).nonzero(as_tuple=True)[0]

    def get_edge_pred_indices(self):
        if hasattr(self, 'edge_query') and self.edge_query is not None:
            return self.edge_query
        return (self.edge_attr == -1).nonzero(as_tuple=True)[0]


class GraphEmbed(torch_geometric.data.Data):
    num_nodes: int
    num_edges: int
    edge_index: torch.tensor  # torch.Size([2, E]), the directed edges
    x: torch.Tensor  # torch.Size([N, F]), the embedding of the entities
    edge_attr: torch.Tensor  # torch.Size([E, F]), the embedding of the edges
    inv_edge_attr: torch.Tensor  # torch.Size([E, F]), the embedding of the inversed edges

    def __init__(self, *args, **kwargs):
        super(GraphEmbed, self).__init__(*args, **kwargs)


# See https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/data/data.html#Data
class DictObject:
    def __getitem__(self, key):
        r"""Gets the data of the attribute :obj:`key`."""
        return getattr(self, key, None)

    def __setitem__(self, key, value):
        """Sets the attribute :obj:`key` to :obj:`value`."""
        setattr(self, key, value)

    @property
    def keys(self):
        r"""Returns all names of graph attributes."""
        keys = [key for key in self.__dict__.keys() if self[key] is not None]
        keys = [key for key in keys if key[:2] != '__' and key[-2:] != '__']
        return keys


class MatGraph:
    x: torch.tensor  # [N], the feature of nodes
    embed_type: torch.tensor  # [N], the type of embedding to be used {Ent: 0, Mask: 1, Rel: 2}
    pred_type: torch.tensor  # [N], the prediction type to be used {Ent: 0, Rel: 1}
    x_query: torch.tensor  # [Q], the node id of queries
    x_ans: torch.tensor  # [Q], optional, the answer of queries
    x_pred_weight: torch.tensor  # [Q]
    attn_bias_type: torch.tensor  # [N, N], the bias of attention (can be used as edge_index)
    pos_x: torch.tensor  # [M], the corresponding node id's (not query id's!)
    pos_ans: torch.tensor  # [M], positive answers that should be exlucded in contrastive loss
    joint_nodes: torch.Tensor  # [2Q], joint queries in u type
    union_query: torch.Tensor  # [Q], the shared nodes in u type

    def __init__(self, x, embed_type, pred_type, x_query=None, x_ans=None, x_pred_weight=None, attn_bias_type=None,
                 pos_x=None, pos_ans=None, joint_nodes=None, union_query=None):
        self.x = x
        self.embed_type = embed_type
        self.pred_type = pred_type
        self.x_query = x_query if x_query is not None else torch.tensor([])
        self.x_ans = x_ans if x_ans is not None else torch.tensor([])
        self.x_pred_weight = x_pred_weight if x_pred_weight is not None else torch.ones(self.x_query.shape[0],
                                                                                        dtype=torch.float)
        self.attn_bias_type = attn_bias_type if attn_bias_type is not None else torch.zeros(
            (self.num_nodes, self.num_nodes), dtype=torch.long)
        self.joint_nodes = joint_nodes if joint_nodes is not None else torch.tensor([])
        self.union_query = union_query if union_query is not None else torch.tensor([])
        assert (pos_ans is None) == (pos_x is None), 'pos_ans and pos_x should be both provided'
        if pos_x is not None:
            assert pos_x.shape[0] >= x_query.shape[0]  # pos_x should include all values in x_query
            self.pos_x = pos_x
            self.pos_ans = pos_ans
        else:
            self.pos_x = x_query
            self.pos_ans = x_ans

    @property
    def num_nodes(self):
        return self.x.shape[0]

    def __getitem__(self, item):
        return self.__getattribute__(item) if hasattr(self, item) else None

    def shallow_clone(self):
        from copy import copy
        return copy(self)

    def pad(self, pad_l):
        from torch.nn.functional import pad
        mg = self.shallow_clone()
        mg.x = pad(self.x, (0, pad_l))
        mg.embed_type = pad(self.embed_type, (0, pad_l))
        mg.pred_type = pad(self.pred_type, (0, pad_l))
        mg.attn_bias_type = pad(self.attn_bias_type, (0, pad_l, 0, pad_l), value=1)
        all_nodes = torch.arange(mg.num_nodes)
        mg.attn_bias_type[all_nodes, all_nodes] = 0
        return mg

    def pad_to(self, n):
        return self.pad(n - self.num_nodes)

    @staticmethod
    def make_line_graph(g: GraphWithAnswer, relation_cnt: int):
        from torch import cat, full, arange
        # Add reverse edges
        tot_n = g.num_nodes + g.num_edges * 2
        edge_index = cat([g.edge_index, g.edge_index[[1, 0]]], dim=1)
        # Features
        edge_attr = cat([g.edge_attr, g.edge_attr + relation_cnt])
        x = cat([g.x, edge_attr])
        embed_type = cat([full(g.x.shape, 0), full(edge_attr.shape, 2)])
        mask = x == -1
        x[mask] = 0
        embed_type[mask] = 1
        # Distinguish nodes and edges
        pred_type = cat([full(g.x.shape, 0), full(edge_attr.shape, 1)])
        # TODO: support edge queries
        # Edges
        attn_bias_type = full((tot_n, tot_n), 1)

        def add_edge(start, end):
            attn_bias_type[end, start] = 0

        all_nodes = arange(tot_n)
        edge_nodes = g.num_nodes + arange(g.num_edges * 2)
        add_edge(all_nodes, all_nodes)  # self-attention
        add_edge(edge_index[0], edge_nodes)
        add_edge(edge_nodes, edge_index[1])
        return MatGraph(
            x=x,
            embed_type=embed_type,
            pred_type=pred_type,
            x_query=g.x_query,
            x_ans=g.x_ans if hasattr(g, 'x_ans') else None,
            x_pred_weight=g.x_pred_weight if hasattr(g, 'x_pred_weight') else None,
            attn_bias_type=attn_bias_type,
            pos_x=g.x_pred_mask[0] if hasattr(g, 'x_pred_mask') else None,
            pos_ans=g.x_pred_mask[1] if hasattr(g, 'x_pred_mask') else None,
            joint_nodes=g.joint_nodes if hasattr(g, 'joint_nodes') else torch.tensor([]),
            union_query=g.union_query if hasattr(g, 'union_query') else torch.tensor([])
        )


class BatchMatGraph(DictObject):
    x: torch.tensor  # [T*N]
    embed_type: torch.tensor  # [T*N]
    pred_type: torch.tensor  # [T*N]
    attn_bias_type: torch.tensor  # [T, N, N]
    x_query: torch.tensor  # [Q] with values in T*N
    x_ans: torch.tensor  # [Q]
    x_pred_weight: torch.tensor  # [Q]
    pos_x: torch.tensor  # [Q] with values in T*N
    pos_ans: torch.tensor  # [Q]
    joint_nodes: torch.tensor  # [2Q]
    union_query: torch.tensor  # [Q]

    @staticmethod
    def from_mat_list(arr: [MatGraph]):
        from torch import cat, stack
        n = max(map(lambda g: g.num_nodes, arr))
        arr = [g.pad_to(n) for g in arr]
        b = BatchMatGraph()
        b.x = cat([g.x for g in arr])
        b.embed_type = cat([g.embed_type for g in arr])
        b.pred_type = cat([g.pred_type for g in arr])
        b.attn_bias_type = stack([g.attn_bias_type for g in arr])
        b.x_query = cat([g.x_query + i * n for i, g in enumerate(arr)])
        b.x_ans = cat([g.x_ans for g in arr])
        b.x_pred_weight = cat([g.x_pred_weight for g in arr])
        b.pos_x = cat([g.pos_x + i * n for i, g in enumerate(arr)])
        b.pos_ans = cat([g.pos_ans for g in arr])
        b.joint_nodes = cat([g.joint_nodes + i * n for i, g in enumerate(arr)])
        b.union_query = cat([g.union_query + i * n for i, g in enumerate(arr)])
        return b

    @property
    def num_graphs(self):
        return self.attn_bias_type.shape[0]

    @property
    def num_nodes_per_graph(self):
        return self.attn_bias_type.shape[1]

    def to(self, device, *keys, **kwargs):
        obj = BatchMatGraph()
        for k in self.keys:
            obj[k] = self[k].to(device, *keys, **kwargs)
        return obj


r"""
Using code from
https://github.com/acbull/LADIES/blob/master/utils.py
"""


def row_normalize_np(mx):
    import numpy as np
    import scipy.sparse as sp
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)

    mx = r_mat_inv.dot(mx)
    return mx


def get_adj_np(edge_index, edge_attr, num_nodes):
    import numpy as np
    import scipy.sparse as sp
    data = edge_attr.numpy()
    row_ind = edge_index[0].numpy()
    col_ind = edge_index[1].numpy()
    return sp.csr_matrix((data, (row_ind, col_ind)),
                         shape=(num_nodes, num_nodes), dtype=np.float32)


def get_directed_lap_matrix_np(edge_index, num_nodes):
    edge_index = to_undirected(edge_index=edge_index, no_dup_self_loops=False)
    import scipy.sparse as sp
    adj = get_adj_np(edge_index, torch.ones(edge_index.shape[1], dtype=torch.float), num_nodes)
    adj = row_normalize_np(adj + sp.eye(num_nodes, format='csr'))
    return adj


def to_undirected(edge_index: torch.Tensor, no_dup_self_loops=True) -> torch.Tensor:
    r"""
    to_undirected() in torch_geometric removes multiple edges
    """
    r_edge_index = edge_index[[1, 0]]
    if no_dup_self_loops:
        mask = edge_index[0] != edge_index[1]
        r_edge_index = r_edge_index[:, mask]
    edge_index = torch.cat([edge_index, r_edge_index], dim=1)
    return edge_index


def get_intv(arr, val):
    int_l = torch.searchsorted(arr, val, right=False)
    int_r = torch.searchsorted(arr, val, right=True)
    return int_l, int_r


class IndexedGraph(Graph):
    def __init__(self, *args, **kwargs):
        super(IndexedGraph, self).__init__(*args, **kwargs)
        edge_index = self.edge_index
        sort_index = edge_index[0] * self.num_nodes + edge_index[1]
        indices = torch.sort(sort_index)[1]
        self.edge_index = self.edge_index[:, indices].contiguous()
        self.edge_attr = self.edge_attr[indices].contiguous()
        sort_index = torch.cat([
            sort_index,
            edge_index[1] * self.num_nodes + edge_index[0],
        ])
        values = torch.unique(sort_index, sorted=True)
        self.adj_index = torch.stack([values // self.num_nodes, values % self.num_nodes])
        # delete self_loop
        self.adj_index = self.adj_index[:, self.adj_index[0] != self.adj_index[1]].contiguous()

    @staticmethod
    def from_graph(graph: Graph) -> 'IndexedGraph':
        return IndexedGraph(**{k: graph[k] for k in graph.keys})

    def get_adj_nodes(self, node):
        l, r = get_intv(self.adj_index[0], torch.tensor([node]))
        l, r = l[0], r[0]
        return self.adj_index[1][l:r]

    def get_induced_subgraph(self, node_list) -> Graph:
        device = node_list.device
        cat_list0 = []
        cat_list1 = []
        cat_list2 = []
        glob_l, glob_r = get_intv(self.edge_index[0], node_list)
        for g_l, g_r in zip(glob_l, glob_r):
            arr_l, arr_r = get_intv(self.edge_index[1][g_l:g_r], node_list)
            for l, r in zip(arr_l, arr_r):
                if l < r:
                    l += g_l
                    r += g_l
                    cat_list0.append(self.edge_index[0, l:r])
                    cat_list1.append(self.edge_index[1, l:r])
                    cat_list2.append(self.edge_attr[l:r])
        if len(cat_list0) == 0:
            return Graph(
                x=self.x,
                edge_index=torch.tensor([[], []], device=device, dtype=torch.long),
                edge_attr=torch.tensor([], device=device, dtype=self.edge_attr.dtype),
            )
        return Graph(
            x=self.x,
            edge_index=torch.stack([torch.cat(cat_list0), torch.cat(cat_list1)]),
            edge_attr=torch.cat(cat_list2),
        )

    def get_edges_between(self, a, b):
        graph = self.get_induced_subgraph(torch.tensor([a, b]))
        self_loop_mask = graph.edge_index[0] != graph.edge_index[1]
        return graph.edge_index[:, self_loop_mask], graph.edge_attr[self_loop_mask]


def relabel_nodes(graph: Graph, node_list: torch.Tensor) -> Graph:
    device = node_list.device
    dfn = {x.item(): i for i, x in enumerate(node_list)}

    return Graph(
        x=graph.x[node_list],
        edge_index=torch.tensor([
            [dfn[x.item()] for x in graph.edge_index[0]],
            [dfn[x.item()] for x in graph.edge_index[1]],
        ], device=device, dtype=torch.long),
        edge_attr=graph.edge_attr,
    )


class EdgeIndexer:
    def __init__(self, num_nodes, relation_cnt):
        self.target_edge = [dict() for i in range(num_nodes)]
        self.relation_cnt = relation_cnt

    def add_edge_no_rev(self, a, p, b):
        d = self.target_edge[a]
        if p in d:
            d[p].append(b)
        else:
            d[p] = [b]

    def add_edge(self, a, p, b):
        self.add_edge_no_rev(a, p, b)
        self.add_edge_no_rev(b, p + self.relation_cnt, a)

    def get_targets(self, a, p):
        return self.target_edge[a].get(p, [])

    def get_rev_targets(self, b, p):
        return self.get_targets(b, p + self.relation_cnt)
