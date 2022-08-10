from typing import Iterable

import torch
from torch.utils.data import Sampler

from graph_util import GraphWithAnswer


class SubsetSumSampler(Sampler):
    def __init__(self, values, lim=512, shuffle=True):
        super(SubsetSumSampler, self).__init__(values)
        n = len(values)
        self._index_list = []
        self._order = list(range(n))
        if shuffle:
            import random
            random.shuffle(self._order)
        cnt = 0
        for i in range(n):
            cur = values[self._order[i]]
            if cnt + cur > lim and cnt > 0:
                cnt = cur
                self._index_list.append(i)
            else:
                cnt += cur
        self._index_list.append(n)

    def __iter__(self):
        l = 0
        for r in self._index_list:
            yield self._order[l:r]
            l = r

    def __getitem__(self, item):
        idx = self._index_list
        l = idx[item - 1] if item != 0 else 0
        r = idx[item]
        return self._order[l:r]

    def __len__(self):
        return len(self._index_list)


def _batch_mini_sampler(batch, igraph, relation_cnt, lap_matrix, config):
    stream = iter(batch)
    from sampler import mini_sampler
    stream = map(
        lambda x: mini_sampler(igraph, x, lap_matrix, relation_cnt, config),
        stream)
    stream = filter(lambda x: x is not None, stream)
    from graph_util import MatGraph, BatchMatGraph
    stream = map(lambda x: MatGraph.make_line_graph(x, relation_cnt), stream)
    g = BatchMatGraph.from_mat_list(list(stream))
    return g


def dataloader_pretrain(full_train_graph, num_nodes, relation_cnt, config) -> Iterable[GraphWithAnswer]:
    r"""
    full_train_graph: A standard Graph object containing all the triples in train.txt
    """
    from graph_util import IndexedGraph, get_directed_lap_matrix_np
    from functools import partial
    igraph = IndexedGraph.from_graph(full_train_graph)
    if config['pretrain_dataset_source'] == 'relation':
        dataset = igraph.edge_index.T
    elif config['pretrain_dataset_source'] == 'entity':
        dataset = torch.arange(num_nodes).unsqueeze(-1)
    else:
        assert False
    # noinspection PyTypeChecker
    return torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        collate_fn=partial(
            _batch_mini_sampler,
            igraph=igraph,
            relation_cnt=relation_cnt,
            lap_matrix=get_directed_lap_matrix_np(igraph.edge_index, igraph.num_nodes),
            config=config,
        ),
        shuffle=True,
    )


def dataloader_test(edge_list, exclude_edges, num_nodes, relation_cnt) -> Iterable[GraphWithAnswer]:
    r"""
    The test edge here is not used in the final metric so it's just another bunch of training data
    edge_list: List of 3-tuples
    exclude_list: List of 3-tuples
    """
    device = torch.device('cpu')
    from graph_util import GraphWithAnswer, EdgeIndexer
    existing_ans = EdgeIndexer(num_nodes, relation_cnt)
    if exclude_edges is not None:
        for a, p, b in exclude_edges:
            existing_ans.add_edge(a, p, b)

    def collate(batch):
        # batch: a list of Tensor[3]
        num_edges = len(batch)

        batch = torch.stack(batch, dim=1)
        r"""
        0   1    2   ... n-1  n  ... 2n-1
        |   |    |   ...  |   |  ...  |
        2n 2n+1 2n+2 ... 3n-1 3n ... 4n-1

        Masked: [n, 3n)
        """
        from graph_util import MatGraph, BatchMatGraph
        g_list = []
        ei_single = torch.tensor([[0], [1]])
        one_arr = torch.tensor([1])
        for i in range(num_edges):
            a, p, b = torch.flatten(batch[:, i]).tolist()
            g = GraphWithAnswer(
                x=torch.tensor([a, -1]),
                edge_index=ei_single,
                edge_attr=torch.tensor([p]),
                x_query=one_arr,
                x_ans=torch.tensor([b]),
            )
            if exclude_edges is not None:
                mask_arr = existing_ans.get_targets(a, p)
                if b not in mask_arr:
                    mask_arr.append(b)
                g.x_pred_mask = torch.tensor([
                    [1] * len(mask_arr),
                    mask_arr,
                ])
            g_list.append(MatGraph.make_line_graph(g, relation_cnt))
        # TODO: another half
        g = BatchMatGraph.from_mat_list(g_list)
        return g

    from torch.utils import data
    dataloader = data.DataLoader(
        torch.tensor(edge_list, device=device),
        batch_size=4096,
        collate_fn=collate,
        shuffle=True,
        num_workers=10,
    )
    return dataloader
