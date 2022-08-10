import random

import numpy as np
import torch

from graph_util import Graph, GraphWithAnswer, IndexedGraph

r"""
edge_index: Tensor of shape [2, E]
"""


def ladies_sampler_np(node_list, samp_num_list, num_nodes, lap_matrix):
    r"""
    LADIES_Sampler: Sample a fixed number of nodes per layer. The sampling probability (importance)
                     is computed adaptively according to the nodes sampled in the upper layer.
    Currently it only returns one single large graph.
    Sample nodes from top to bottom, based on the probability computed adaptively (layer-dependent).
    References:
    * https://github.com/acbull/LADIES/blob/master/pytorch_ladies.py
    """
    for d in samp_num_list:
        # Row-select the lap_matrix (U) by previously sampled nodes
        u = lap_matrix[node_list, :]
        # Only use the upper layer's neighborhood to calculate the probability.
        pi = np.array(np.sum(u.multiply(u), axis=0))[0]
        del u
        # What's chosen should not be considered again
        pi[node_list] = 0
        s_num = np.min([np.sum(pi > 0), d])
        if s_num == 0:
            break
        pi = pi / np.sum(pi)
        # Sample the next layer's nodes based on the adaptively probability (p).
        after_nodes = np.random.choice(num_nodes, s_num, p=pi, replace=False)
        node_list = np.unique(np.concatenate((node_list, after_nodes)))
        del after_nodes

    return node_list


def sample_meta_tree(origin_node, igraph: IndexedGraph, edge_add_ratio=0.8) -> Graph:
    device = igraph.edge_index.device
    dfn = {origin_node: 0}
    idx = [origin_node]
    import numpy as np
    num_iter = np.random.randint(5, 16)
    ei0 = []
    ei1 = []
    ea = []
    for i in range(num_iter):
        c = np.random.choice(idx)
        adj_node_list = igraph.get_adj_nodes(c)
        if len(adj_node_list) == 0:
            break
        adj_node = np.random.choice(adj_node_list)
        adj_ei, adj_ea = igraph.get_edges_between(c, adj_node)
        e_id = np.random.choice(len(adj_ea))
        a, p, b = adj_ei[0][e_id].item(), adj_ea[e_id].item(), adj_ei[1][e_id].item()
        assert a == c or b == c
        if a in dfn and b in dfn:
            continue
        if a not in dfn:
            idx.append(a)
            dfn[a] = len(dfn)
        if b not in dfn:
            idx.append(b)
            dfn[b] = len(dfn)
        ei0.append(a)
        ei1.append(b)
        ea.append(p)
    g = Graph(
        x=igraph.x,
        edge_index=torch.tensor([ei0, ei1], dtype=torch.long),
        edge_attr=torch.tensor(ea, dtype=torch.long),
    )
    idx = torch.tensor(idx, dtype=torch.long, device=device)
    g = sample_induced_edges(igraph, g, idx, edge_add_ratio)
    from graph_util import relabel_nodes
    g = relabel_nodes(g, idx)
    return g


def sample_from_origin(node_list, igraph: IndexedGraph, lap_matrix, ladies_size, induced_edge_prob=1.0):
    device = igraph.edge_index.device
    num_nodes = igraph.num_nodes

    # Sampling around the nodes to be predicted
    import random
    def rand_partition(n_ele, n_parts):
        assert n_parts <= n_ele
        from random import randint
        if n_ele == 0:
            return []
        bar = [randint(0, n_ele - n_parts) for _ in range(n_parts - 1)]
        bar += [0, n_ele - n_parts]
        bar.sort()
        return [bar[i] - bar[i - 1] + 1 for i in range(1, n_parts + 1)]

    n_ele = random.randint(1, ladies_size)
    n_parts = random.randint(0, n_ele)
    node_list = ladies_sampler_np(
        node_list.numpy(), rand_partition(n_ele, n_parts),
        num_nodes, lap_matrix)
    node_list = torch.from_numpy(node_list).to(device).contiguous()
    graph = Graph(
        x=igraph.x,
        edge_index=torch.tensor([[], []], dtype=torch.long, device=device),
        edge_attr=torch.tensor([], dtype=torch.long, device=device),
    )
    graph = sample_induced_edges(igraph, graph, node_list, induced_edge_prob)
    from graph_util import relabel_nodes
    graph = relabel_nodes(graph, node_list)
    return graph


def gen_ans_by_masking(graph: Graph, num_nodes, relation_cnt, config) -> GraphWithAnswer:
    device = graph.edge_index.device
    mask_ratio = config['pretrain_mask_ratio']
    mask_type_ratio = config['pretrain_mask_type_ratio']

    # Choose the nodes to predict again
    def mask_num(size):
        import random
        assert size >= 0
        size_map = {
            0: [0, 0],
            1: [1, 1],  # Edge only!
            2: [1, 1],
            3: [1, 1],
            4: [1, 2],
            5: [1, 2],
            6: [1, 3],
            7: [2, 3],
            8: [2, 3],
            9: [2, 3],
            10: [2, 4],
            11: [2, 4],
            12: [3, 4],
            13: [3, 4],
            14: [3, 5],
            15: [3, 5],
            16: [4, 5],
            17: [4, 6],
            18: [4, 6],
            19: [5, 7],
            20: [5, 7],
            21: [6, 7],
            22: [6, 7],
            23: [6, 8],
            24: [6, 8],
            25: [7, 8],
            26: [7, 9],
            27: [7, 9],
            28: [8, 9],
            29: [8, 9],
            30: [8, 10],
            31: [8, 10],
            32: [9, 10]
        }
        from math import floor
        return random.randint(*size_map.get(size, [floor(size * mask_ratio[0]), floor(size * mask_ratio[1])]))

    from math import floor
    nmask = floor((mask_type_ratio[0] / (mask_type_ratio[0] + mask_type_ratio[1])) * graph.num_nodes)
    emask = floor(
        (mask_type_ratio[1] / (mask_type_ratio[0] + mask_type_ratio[1])) * graph.num_nodes)  # still use node_size here
    emask = min(emask, graph.num_edges)
    num_node_mask = mask_num(nmask)
    num_edge_mask = mask_num(emask)
    x_query = torch.from_numpy(np.random.choice(graph.num_nodes, num_node_mask, replace=False)).to(device)
    edge_query = torch.from_numpy(np.random.choice(graph.num_edges, num_edge_mask, replace=False)).to(device)

    x_ans = graph.x[x_query]
    edge_ans = graph.edge_attr[edge_query]

    def do_mask(arr, list_predict, num_nodes) -> torch.LongTensor:
        randseed = random.random()
        if randseed < config['mask_ratio']:
            arr[list_predict] = -1
        elif config['mask_ratio'] <= randseed < (config['mask_ratio'] + 1) / 2:
            arr[list_predict] = torch.randint(low=0, high=num_nodes, size=list_predict.shape, device=device)
        else:
            pass
        return arr

    return GraphWithAnswer(
        edge_index=graph.edge_index,
        x=do_mask(graph.x, x_query, num_nodes),
        edge_attr=do_mask(graph.edge_attr, edge_query, relation_cnt),
        x_query=x_query,
        x_ans=x_ans,
        edge_query=edge_query,
        edge_ans=edge_ans,
        joint_nodes=torch.tensor([], device=device, dtype=torch.long),
        union_query=torch.tensor([], device=device, dtype=torch.long)
    )


def sample_induced_edges(igraph: IndexedGraph, graph: Graph, node_list, add_prob) -> Graph:
    subg = igraph.get_induced_subgraph(node_list)
    existing_edge = set()
    for a, b, p in zip(graph.edge_index[0], graph.edge_index[1], graph.edge_attr):
        existing_edge.add((a.item(), b.item(), p.item()))
    import random
    idx = []
    for i in range(subg.num_edges):
        if random.random() > add_prob:
            continue
        if (subg.edge_index[0][i].item(), subg.edge_index[1][i].item(), subg.edge_attr[i].item()) in existing_edge:
            continue
        idx.append(i)
    newg = Graph(
        x=graph.x,
        edge_index=torch.cat([graph.edge_index, subg.edge_index[:, idx]], dim=1),
        edge_attr=torch.cat([graph.edge_attr, subg.edge_attr[idx]]),
    )
    return newg


def sample_n_p(igraph: IndexedGraph, target_node, n, config):
    dfn = {target_node: 0}
    idx = [target_node]
    cur_node = target_node
    edge_index0 = []
    edge_index1 = []
    edge_attr = []
    sampled_node = [target_node]
    for i in range(n):
        adj_nodes = list(igraph.get_adj_nodes(cur_node))
        if not adj_nodes:
            return None
        nxt_node = np.random.choice(adj_nodes)
        dfn[nxt_node] = len(dfn)
        idx.append(nxt_node)
        ei, ea = igraph.get_edges_between(cur_node, nxt_node)
        ei = list(ei)
        ea = list(ea)
        p = np.random.choice(len(ea))
        e0 = ei[0][p]
        e1 = ei[1][p]
        if e0 != nxt_node and e1 != nxt_node:
            while True:
                p = np.random.choice(len(ea))
                e0 = ei[0][p]
                e1 = ei[1][p]
                if e0 == nxt_node or e1 == nxt_node:
                    break
        if e0 == idx[-1]:
            edge_index0.append(len(idx) - 1)
            edge_index1.append(len(idx) - 2)
        elif e1 == idx[-1]:
            edge_index0.append(len(idx) - 2)
            edge_index1.append(len(idx) - 1)
        else:
            assert False
        edge_attr.append(ea[p])
        sampled_node.append(nxt_node)
        cur_node = nxt_node
    result = Graph(
        x=igraph.x[idx],
        edge_index=torch.tensor([edge_index0, edge_index1]),
        edge_attr=torch.tensor(edge_attr),
    )
    idx = torch.tensor(idx)
    randseed = random.random()
    if randseed <= config['p_mask_ratio']:
        x_node = torch.tensor([-1] * n + [idx[-1]])
        ans = torch.tensor([target_node])
    elif config['p_mask_ratio'] < randseed <= (config['mask_ratio'] + 1) / 2:
        x_node = torch.tensor([-1] * n + [idx[-1]])
        ans = torch.randint(low=0, high=igraph.num_nodes, size=(1,), dtype=torch.long)
    else:
        x_node = torch.tensor([target_node] + [-1] * (n - 1) + [idx[-1]])
        ans = torch.tensor([target_node])
    result = GraphWithAnswer(
        x=x_node,
        edge_index=result.edge_index,
        edge_attr=result.edge_attr,
        x_query=torch.tensor([0], dtype=torch.long),
        x_ans=ans,
        edge_query=torch.tensor([], dtype=torch.long),
        edge_ans=torch.tensor([], dtype=torch.long),
        joint_nodes=torch.tensor([], dtype=torch.long),
        union_query=torch.tensor([], dtype=torch.long)
    )
    return result


def sample_one_edge_between(igraph, a, b):
    ei, ea = igraph.get_edges_between(a, b)
    ptr = np.random.choice(len(ea))
    return ei[0, ptr], ei[1, ptr], ea[ptr]


def sample_n_i(igraph: IndexedGraph, target_node, n, edge_drop_out_rate):
    full_adj_node = igraph.get_adj_nodes(target_node)
    if len(full_adj_node) < n:
        return None
    adj_node = np.random.choice(full_adj_node, size=n, replace=False).tolist()
    edge_index0 = []
    edge_index1 = []
    edge_attr = []
    idx = [target_node] + list(adj_node)
    for ind, i in enumerate(adj_node):
        a, b, p = sample_one_edge_between(igraph, target_node, i)
        if a != target_node and b != target_node:
            while True:
                a, b, p = sample_one_edge_between(igraph, target_node, i)
                if a == target_node or b == target_node:
                    break
        if a == target_node:
            edge_index0.append(0)
            edge_index1.append(ind + 1)
        elif b == target_node:
            edge_index0.append(ind + 1)
            edge_index1.append(0)
        else:
            assert False
        edge_attr.append(p)
    g = Graph(
        x=igraph.x,
        edge_index=torch.tensor([edge_index0, edge_index1], dtype=torch.long),
        edge_attr=torch.tensor(edge_attr),
    )
    g = GraphWithAnswer(
        x=torch.tensor([-1] + adj_node),
        edge_index=g.edge_index,
        edge_attr=g.edge_attr,
        x_query=torch.tensor([0], dtype=torch.long),
        x_ans=torch.tensor([target_node], dtype=torch.long),
        edge_query=torch.tensor([], dtype=torch.long),
        edge_ans=torch.tensor([], dtype=torch.long),
        joint_nodes=torch.tensor([], dtype=torch.long),
        union_query=torch.tensor([], dtype=torch.long)
    )
    return g


def cap_edges(g: GraphWithAnswer, max_num_edges):
    if g.num_edges <= max_num_edges:
        return g
    g = g.clone()
    new_edges = torch.multinomial(torch.ones([g.num_edges]), max_num_edges, replacement=False)
    chosen = torch.zeros([g.num_edges], dtype=torch.bool)
    chosen[new_edges] = True
    g.edge_index = g.edge_index[:, chosen]
    g.edge_attr = g.edge_attr[chosen]
    if hasattr(g, 'edge_query'):
        q_mask = chosen[g.edge_query]
        g.edge_query = g.edge_query[q_mask]
        g.edge_ans = g.edge_ans[q_mask]
    return g


def mini_sampler(igraph: IndexedGraph, target_nodes, lap_matrix, relation_cnt, config) -> GraphWithAnswer:
    target_node = np.random.choice(target_nodes)
    d: dict = config['pretrain_sampler_ratio']
    edge_drop_out_rate = config['edge_drop_out_rate']
    a = list(d.keys())
    p = np.array(list(map(d.__getitem__, a)), dtype=np.float32)
    p = p / p.sum()
    g_type = np.random.choice(a=a, p=p)
    g = None
    for n in range(config['sample_retries']):
        if g_type == '1p':
            g = sample_n_p(igraph, target_node, 1, config)
        elif g_type == '2p':
            g = sample_n_p(igraph, target_node, 2, config)
        elif g_type == '3p':
            g = sample_n_p(igraph, target_node, 3, config)
        elif g_type == '2i':
            g = sample_n_i(igraph, target_node, 2, edge_drop_out_rate)
        elif g_type == '3i':
            g = sample_n_i(igraph, target_node, 3, edge_drop_out_rate)
        elif g_type == 'meta_tree':
            g = sample_meta_tree(target_node, igraph, edge_add_ratio=config['induced_edge_prob'])
        elif g_type == 'ladies':
            g = sample_from_origin(
                node_list=target_nodes,
                igraph=igraph,
                lap_matrix=lap_matrix,
                ladies_size=config['ladies_size'],
                induced_edge_prob=config['induced_edge_prob'],
            )
        else:
            raise NotImplementedError(f'Sample type "{g_type}" not implemented')
        if g is not None:
            break
    if isinstance(g, Graph) and not isinstance(g, GraphWithAnswer):
        g = gen_ans_by_masking(
            graph=g,
            num_nodes=igraph.num_nodes,
            relation_cnt=relation_cnt,
            config=config,
        )
    if g is not None:
        assert isinstance(g, GraphWithAnswer)
        g = cap_edges(g, 50)
    return g
