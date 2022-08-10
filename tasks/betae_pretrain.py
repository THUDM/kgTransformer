import torch

from graph_util import Graph


class FB15K237:
    def __init__(self, config):
        from .betae import BetaEDataset
        self.betae = BetaEDataset(config['data_dir'])
        self.train_edge, self.valid_edge, self.test_edge = self.betae.calldata()
        self.num_nodes = 0
        self.relation_cnt = 0

        def edge_sanitize(a, r, b):
            if r % 2 == 1:
                a, b = b, a
                r = int((r - 1) / 2)
                return None  # The edge is added in the inverse counterpart
            else:
                r = int(r / 2)
            if a >= self.num_nodes:
                self.num_nodes = a + 1
            if b >= self.num_nodes:
                self.num_nodes = b + 1
            if r >= self.relation_cnt:
                self.relation_cnt = r + 1
            return a, r, b

        def batch_edge_san(arr):
            for t in arr:
                t = edge_sanitize(*t)
                if t is not None:
                    yield t

        self.train_edge = list(batch_edge_san(self.train_edge))
        self.valid_edge = list(batch_edge_san(self.valid_edge))
        self.test_edge = list(batch_edge_san(self.test_edge))

    def get_full_train_graph(self):
        device = torch.device('cpu')
        arr = torch.tensor(self.train_edge, device=device).T
        return Graph(
            x=torch.arange(self.num_nodes, device=device),
            edge_index=arr[[0, 2]],
            edge_attr=arr[1],
        )

    def dataloader_train(self, config):
        from data_util import dataloader_pretrain
        return dataloader_pretrain(
            self.get_full_train_graph(),
            self.num_nodes,
            self.relation_cnt,
            config,
        )

    def dataloader_test(self):
        from data_util import dataloader_test
        return dataloader_test(
            self.test_edge,
            self.train_edge + self.valid_edge + self.test_edge,
            self.num_nodes,
            self.relation_cnt,
        )
