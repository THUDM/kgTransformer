import torch

from graph_util import GraphWithAnswer


def query_to_graph(mode, q, hard_ans_list, mask_list=None, gen_pred_mask=True, device=torch.device('cpu')):
    r"""
    Assert that easy_ans_set have included all nodes to be masked
    """
    if not gen_pred_mask:
        assert mask_list is None or len(hard_ans_list) == len(mask_list)
    x = []
    edge_ans = []

    edge_index = [[], []]
    edge_attr = []

    def add_raw_edge(a, r, b):
        if r % 2 == 1:
            a, b = b, a
            r = (r - 1) / 2
        else:
            r = r / 2
        edge_index[0].append(a)
        edge_index[1].append(b)
        edge_attr.append(r)

    q_cnt = 0
    x_query = []
    x_ans = []
    x_pred_weight = []
    x_pred_mask_x = []
    x_pred_mask_y = []
    joint_nodes = []
    union_query = []

    def push_anslist(node_id, hard_anslist, mask_list):
        nonlocal x_query, x_ans, x_pred_weight
        anslen = len(hard_anslist)
        if anslen == 0:
            return
        x_query += [node_id] * anslen
        x_ans += hard_anslist
        x_pred_weight += [1 / anslen] * anslen
        if not gen_pred_mask:
            return
        nonlocal x_pred_mask_x, x_pred_mask_y, q_cnt
        # assert that mask_list already contains hard_anslist
        x_pred_mask_x += [node_id] * len(mask_list)
        x_pred_mask_y += mask_list

    def push_anslist_and_masklist(node_id, mask_id, hard_anslist, mask_list):
        nonlocal x_query, x_ans, x_pred_weight
        anslen = len(hard_anslist)
        if anslen == 0:
            return
        x_query += [node_id] * anslen
        x_ans += hard_anslist
        x_pred_weight += [1 / anslen] * anslen
        if not gen_pred_mask:
            return
        nonlocal x_pred_mask_x, x_pred_mask_y, q_cnt
        for i in mask_id:
            x_pred_mask_x += [i] * len(mask_list)
            x_pred_mask_y += mask_list

    mask_list = mask_list or []
    if mode == '1p':
        x = [q[0], -1]
        add_raw_edge(0, q[1][0], 1)
        push_anslist(1, hard_ans_list, mask_list)
    elif mode == '2p':
        r"""
        0 - 1(-1) - 2(-1)
        """
        x = [q[0], -1, -1]
        add_raw_edge(0, q[1][0], 1)
        add_raw_edge(1, q[1][1], 2)
        push_anslist(2, hard_ans_list, mask_list)
    elif mode == '3p':
        r"""
        0 - 1(-1) - 2(-1) - 3(-1)
        """
        x = [q[0], -1, -1, -1]
        add_raw_edge(0, q[1][0], 1)
        add_raw_edge(1, q[1][1], 2)
        add_raw_edge(2, q[1][2], 3)
        push_anslist(3, hard_ans_list, mask_list)
    elif mode == '2i':
        r"""
        0 - 
            2(-1)
        1 - 
        """
        x = [q[0][0], q[1][0], -1]
        add_raw_edge(0, q[0][1][0], 2)
        add_raw_edge(1, q[1][1][0], 2)
        push_anslist(2, hard_ans_list, mask_list)
    elif mode == '3i':
        r"""
        0 - 
        1 -  3(-1)
        2 - 
        """
        x = [q[0][0], q[1][0], q[2][0], -1]
        add_raw_edge(0, q[0][1][0], 3)
        add_raw_edge(1, q[1][1][0], 3)
        add_raw_edge(2, q[2][1][0], 3)
        push_anslist(3, hard_ans_list, mask_list)
    elif mode == 'pi':
        r"""
        0 - 1(-1) -
                    3(-1)
                2 -  
        """
        x = [q[0][0], -1, q[1][0], -1]
        add_raw_edge(0, q[0][1][0], 1)
        add_raw_edge(1, q[0][1][1], 3)
        add_raw_edge(2, q[1][1][0], 3)
        push_anslist(3, hard_ans_list, mask_list)
    elif mode == 'ip':
        r"""
        0 - 
            2(-1) - 3(-1)
        1 - 
        """
        x = [q[0][0][0], q[0][1][0], -1, -1]
        add_raw_edge(0, q[0][0][1][0], 2)
        add_raw_edge(1, q[0][1][1][0], 2)
        add_raw_edge(2, q[1][0], 3)
        push_anslist(3, hard_ans_list, mask_list)
    elif mode == '2u':
        r"""
                0 - 2(-1)
                |
                4(-1)
                |
                1 - 3(-1)
        """
        x = [q[0][0], q[1][0], -1, -1, -1]
        add_raw_edge(0, q[0][1][0], 2)
        add_raw_edge(0, q[0][1][0], 4)
        add_raw_edge(1, q[1][1][0], 3)
        add_raw_edge(1, q[1][1][0], 4)
        push_anslist_and_masklist(4, [2, 3], hard_ans_list, mask_list)
        anslen = len(hard_ans_list)
        joint_nodes = [2, 3] * anslen
        union_query = [4] * anslen
    elif mode == 'up':
        r"""
                0 - 2(-1) - 3(-1)
                |
                6(-1) - 7(-1)
                |
                1 - 4(-1) - 5(-1)
        """
        x = [q[0][0][0], q[0][1][0], -1, -1, -1, -1, -1, -1]
        add_raw_edge(0, q[0][0][1][0], 2)
        add_raw_edge(0, q[0][0][1][0], 6)
        add_raw_edge(1, q[0][1][1][0], 4)
        add_raw_edge(1, q[0][1][1][0], 6)
        add_raw_edge(2, q[1][0], 3)
        add_raw_edge(6, q[1][0], 7)
        add_raw_edge(4, q[1][0], 5)
        push_anslist_and_masklist(7, [3, 5], hard_ans_list, mask_list)
        anslen = len(hard_ans_list)
        joint_nodes = [3, 5] * anslen
        union_query = [7] * anslen
    else:
        assert False
    g = GraphWithAnswer(
        x=torch.tensor(x, device=device, dtype=torch.long),
        edge_index=torch.tensor(edge_index, device=device, dtype=torch.long),
        edge_attr=torch.tensor(edge_attr, device=device, dtype=torch.long),
        x_query=torch.tensor(x_query, device=device, dtype=torch.long),
        x_ans=torch.tensor(x_ans, device=device, dtype=torch.long),
        edge_ans=torch.tensor(edge_ans, device=device, dtype=torch.long),
        x_pred_weight=torch.tensor(x_pred_weight, device=device, dtype=torch.float),
        joint_nodes=torch.tensor(joint_nodes, device=device, dtype=torch.long),
        union_query=torch.tensor(union_query, device=device, dtype=torch.long)
    )
    if gen_pred_mask:
        g.x_pred_mask = torch.tensor([x_pred_mask_x, x_pred_mask_y], device=device, dtype=torch.long)
    return g


from .betae import BetaEDataset


class FB15K237_reasoning:
    def __init__(self, betae: BetaEDataset, relation_cnt, train_mode, test_mode):
        super(FB15K237_reasoning, self).__init__()
        self.betae = betae
        self.train_mode = train_mode
        self.test_mode = test_mode
        self.relation_cnt = relation_cnt

        self.train_query = betae.get_file("train-queries.pkl")
        self.valid_query = betae.get_file("valid-queries.pkl")
        self.test_query = betae.get_file("test-queries.pkl")

        self.train_answer = betae.get_file("train-answers.pkl")
        self.valid_hard_answer = betae.get_file("valid-hard-answers.pkl")
        self.valid_easy_answer = betae.get_file("valid-easy-answers.pkl")
        self.valid_answer = betae.get_file("valid-answers.pkl")
        self.test_hard_answer = betae.get_file("test-hard-answers.pkl")
        self.test_easy_answer = betae.get_file("test-easy-answers.pkl")
        self.test_answer = betae.get_file("test-answers.pkl")

    @staticmethod
    def _batch_q2g(batch, relation_cnt):
        cpu = torch.device('cpu')
        from graph_util import MatGraph, BatchMatGraph
        arr = [query_to_graph(*x, device=cpu) for x in batch]
        arr = [MatGraph.make_line_graph(g, relation_cnt) for g in arr]
        return BatchMatGraph.from_mat_list(arr)

    def _get_test_dataloader(self, query, pred_answer, mask_answer, modelist, gen_pred_mask, batch_size,
                             num_workers):
        data = []
        for m in modelist:
            for q in query[m]:
                mask_set = pred_answer[q]
                if mask_answer is not None:
                    mask_set = mask_set | mask_answer[q]
                data.append((m, q, list(pred_answer[q]), list(mask_set), gen_pred_mask))
        from torch.utils.data import DataLoader
        from data_util import SubsetSumSampler
        from functools import partial
        data = DataLoader(
            data,
            batch_sampler=SubsetSumSampler(
                [len(x[2]) for x in data],
                lim=batch_size,
            ),
            collate_fn=partial(
                FB15K237_reasoning._batch_q2g,
                relation_cnt=self.relation_cnt,
            ),
            num_workers=num_workers,
        )
        return data

    def dataloader_fine_tune(self, config):
        mode = self.train_mode
        ans = self.train_answer
        return self._get_test_dataloader(self.train_query, pred_answer=ans, mask_answer=None,
                                         modelist=mode, gen_pred_mask=False,
                                         batch_size=config['batch_size'],
                                         num_workers=config['num_workers'])

    def dataloader_test(self, config):
        mode = self.test_mode
        hardans = self.test_hard_answer
        easyans = self.test_easy_answer
        query = self.test_query
        return self._get_test_dataloader(query, pred_answer=hardans, mask_answer=easyans, modelist=mode,
                                         gen_pred_mask=True, batch_size=config['batch_size'],
                                         num_workers=config['num_workers'])

    def dataloader_valid(self, config):
        mode = self.test_mode
        hardans = self.valid_hard_answer
        easyans = self.valid_easy_answer
        query = self.valid_query
        return self._get_test_dataloader(query, pred_answer=hardans, mask_answer=easyans, modelist=mode,
                                         gen_pred_mask=True, batch_size=config['batch_size'],
                                         num_workers=config['num_workers'])
