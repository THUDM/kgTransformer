import enum
from typing import Callable, List, Union

import torch
import torch.nn.functional as F
from fmoe import FMoETransformerMLP
from torch import nn
from torch.nn import Module

from graph_util import Graph, GraphEmbed, GraphWithAnswer, BatchMatGraph


class SparseMultiheadAttention(Module):
    r"""Sparse version of torch.nn.MultiheadAttention
    References:
    * https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html
    """

    def __init__(self, embed_dim, num_heads, dropout=0., bias=True, add_bias_kv=False,
                 add_zero_attn=False, kdim=None, vdim=None):
        super(SparseMultiheadAttention, self).__init__()

        self.embed_dim = embed_dim
        self.kdim = kdim or embed_dim
        self.vdim = vdim or embed_dim

        self.num_heads = num_heads
        self.dropout = torch.nn.Dropout(p=dropout, inplace=False)
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        self.q_proj = torch.nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = torch.nn.Linear(self.kdim, embed_dim, bias=add_bias_kv)
        self.v_proj = torch.nn.Linear(self.vdim, embed_dim, bias=add_bias_kv)
        self.out_proj = torch.nn.Linear(embed_dim, embed_dim, bias=True)
        self.add_zero_attn = add_zero_attn

        self._reset_parameters()

    def _reset_parameters(self):
        from torch.nn.init import xavier_uniform_, xavier_normal_, constant_
        xavier_uniform_(self.q_proj.weight)
        xavier_uniform_(self.k_proj.weight)
        xavier_uniform_(self.v_proj.weight)

        if self.q_proj.bias is not None:
            constant_(self.q_proj.bias, 0.)
        if self.out_proj.bias is not None:
            constant_(self.out_proj.bias, 0.)
        # Note that the init for {k_proj,v_proj}.bias is not the same as above
        # See https://pytorch.org/docs/master/_modules/torch/nn/modules/activation.html#MultiheadAttention
        if self.k_proj.bias is not None:
            xavier_normal_(self.k_proj.bias)
        if self.v_proj.bias is not None:
            xavier_normal_(self.v_proj.bias)

    def forward(self, query, key, value, edge_index, need_weights=True):
        r"""
        :param query: Tensor, shape [tgt_len, embed_dim]
        :param key: Tensor of shape [src_len, kdim]
        :param value: Tensor of shape [src_len, vdim]
        :param edge_index: Tensor of shape [2, E], a sparse matrix that has shape len(query)*len(key),
        :param need_weights: if True, also returns a Tensor of shape [E] that represents the average attention weight
        :return Tensor of shape [tgt_len, embed_dim]
        Reference:
        * https://github.com/pytorch/pytorch/blob/master/torch/nn/functional.py -> multi_head_attention_forward()
        """

        # Dimension checks
        assert edge_index.shape[0] == 2
        assert key.shape[0] == value.shape[0]
        # Dictionary size
        src_len, tgt_len, idx_len = key.shape[0], query.shape[0], edge_index.shape[1]

        scaling = float(self.head_dim) ** -0.5

        assert query.shape[1] == self.embed_dim
        q: torch.Tensor = self.q_proj(query) * scaling
        assert key.shape[1] == self.kdim
        k: torch.Tensor = self.k_proj(key)
        assert value.shape[1] == self.vdim
        v: torch.Tensor = self.v_proj(value)
        assert self.embed_dim == q.shape[1] == k.shape[1] == v.shape[1]

        # Split into heads
        q = q.contiguous().view(tgt_len, self.num_heads, self.head_dim)
        k = k.contiguous().view(src_len, self.num_heads, self.head_dim)
        v = v.contiguous().view(src_len, self.num_heads, self.head_dim)

        # Get score
        attn_output_weights = (torch.index_select(q, 0, edge_index[0]) * torch.index_select(k, 0, edge_index[1])).sum(
            dim=-1)
        assert list(attn_output_weights.size()) == [idx_len, self.num_heads]

        from deter_util import det_softmax
        attn_output_weights = det_softmax(src=attn_output_weights, index=edge_index[0], num_nodes=tgt_len)
        attn_output_weights = self.dropout(attn_output_weights)

        """ Get values """
        attn_output = attn_output_weights.unsqueeze(2) * torch.index_select(v, 0, edge_index[1])
        """
        Aggregation
        References:
        * https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.message_passing.MessagePassing.aggregate
        """
        from deter_util import deter_scatter_add_
        attn_output = deter_scatter_add_(edge_index[0], attn_output,
                                         torch.zeros((tgt_len, attn_output.shape[1], attn_output.shape[2]),
                                                     device=attn_output.device))
        assert list(attn_output.size()) == [tgt_len, self.num_heads, self.head_dim]
        attn_output = self.out_proj(attn_output.contiguous().view(tgt_len, self.embed_dim))
        assert list(attn_output.size()) == [tgt_len, self.embed_dim]

        # average attention weights over heads
        return attn_output, attn_output_weights.mean(dim=1) if need_weights else None


class CustomizedMoEPositionwiseFF(FMoETransformerMLP):
    def __init__(self, d_model, d_inner, dropout, pre_lnorm=False, sandwich_lnorm=False, moe_num_expert=64,
                 moe_top_k=2):
        activation = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        super().__init__(num_expert=moe_num_expert, d_model=d_model, d_hidden=d_inner, top_k=moe_top_k,
                         activation=activation)

        self.pre_lnorm = pre_lnorm
        self.sandwich_lnorm = sandwich_lnorm
        self.batch_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inp):
        if self.pre_lnorm:
            ##### layer normalization + positionwise feed-forward
            core_out = super().forward(self.batch_norm(inp))
            core_out = self.dropout(core_out)

            ##### residual connection
            output = core_out + inp
        elif self.sandwich_lnorm:
            ##### normalization + positionwise feed-forward + normalization
            core_out = super().forward(self.batch_norm(inp))
            core_out = self.batch_norm(core_out)
            core_out = self.dropout(core_out)

            ##### residual connection
            output = core_out + inp
        else:
            ##### positionwise feed-forward
            core_out = super().forward(inp)
            core_out = self.dropout(core_out)

            ##### residual connection + batch normalization
            output = self.batch_norm(inp + core_out)

        return output


class GraphEncoderLayer(Module):
    r"""
    Similar to GAT but using torch.nn.MultiheadAttention
    References:
        * https://pytorch.org/docs/stable/generated/torch.nn.TransformerEncoderLayer.html
        * https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/transformer.py
        * https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.GATConv
    """

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation: Callable = torch.nn.LeakyReLU(),
                 moe=False, moe_num_expert=8, moe_top_k=2):
        super(GraphEncoderLayer, self).__init__()
        self.attn = SparseMultiheadAttention(d_model, num_heads=nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.moe = moe
        if self.moe:
            assert dim_feedforward % moe_top_k == 0
            self.ff_layer = CustomizedMoEPositionwiseFF(d_model, dim_feedforward // moe_top_k, dropout, False, False,
                                                        moe_num_expert, moe_top_k)
        else:
            self.linear1 = torch.nn.Linear(d_model, dim_feedforward)
            self.dropout = torch.nn.Dropout(dropout)
            self.linear2 = torch.nn.Linear(dim_feedforward, d_model)
            self.dropout2 = torch.nn.Dropout(dropout)
            self.norm2 = torch.nn.BatchNorm1d(d_model)
            self.activation = activation

        self.norm1 = torch.nn.BatchNorm1d(d_model)
        self.dropout1 = torch.nn.Dropout(dropout)

    def forward(self, src: torch.Tensor, edge_index: torch.Tensor, add_self_loops: bool = True) -> torch.Tensor:
        r"""
        :param src: Tensor of shape [N, d_model]
        :param edge_index: Tensor of shape [2,E], (a, b) means flowing information from a to b
        :param add_self_loops: bool
        """
        num_nodes = src.shape[0]
        # Allow each node to pay attention to itself
        if add_self_loops:
            from torch_geometric.utils import add_self_loops
            edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
        # This need to be transposed since attn takes the sparse matrix of form [query, key]
        src2 = self.attn(src, src, src, torch.stack([
            edge_index[1], edge_index[0],
        ], dim=0))[0]
        src = src + self.dropout1(src2)  # Residual 1
        src = self.norm1(src)

        if self.moe:
            src = self.ff_layer(src)
        else:
            src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
            src = src + self.dropout2(src2)  # Residual 2
            src = self.norm2(src)
        return src


class GraphTransformer(Module):
    r"""
    References:
    * https://pytorch.org/docs/stable/generated/torch.nn.Transformer.html
    * https://pytorch.org/docs/stable/generated/torch.nn.TransformerEncoder.html
    * https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    * "Attention Is All You Need"
    """

    def __init__(self, d_model: int = 512, nhead: int = 8, num_encoder_layers: int = 6,
                 dim_feedforward: int = 2048, dropout: float = 0.1, activation: Callable = None,
                 moe: bool = False, moe_num_expert: int = 8, moe_top_k: int = 2) -> None:
        super(GraphTransformer, self).__init__()
        from torch.nn import ModuleList
        self.encoder_layers = ModuleList([
            GraphEncoderLayer(d_model, nhead, dim_feedforward=dim_feedforward, dropout=dropout, activation=activation,
                              moe=moe, moe_num_expert=moe_num_expert, moe_top_k=moe_top_k)
            for i in range(num_encoder_layers)
        ])
        self.encoder_norm = torch.nn.LayerNorm(d_model)
        self._reset_parameters()
        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""
        from torch.nn import init
        for p in self.parameters():
            if p.dim() > 1:
                init.xavier_uniform_(p)

    def forward(self, src, edge_index: torch.Tensor) -> torch.Tensor:
        if src.shape[1] != self.d_model:
            raise RuntimeError("the feature number of src must be equal to d_model")

        # Encode
        memory = src
        for mod in self.encoder_layers:
            memory = mod(memory, edge_index)
        memory = self.encoder_norm(memory)

        return memory


class ToTripleGraph:
    r"""
    This is not a module since it has no parameters.
    """

    def __init__(self, preserve_nodes: bool = True, directed=True):
        r"""
        :param preserve_nodes: if false, the "edge" nodes will be directly connected
        """
        # Not preserving nodes are not yet supported
        assert preserve_nodes
        self.preserve_node = preserve_nodes
        # Undirected graphs are not yet supported
        assert directed

    def __call__(self, data: Graph) -> Graph:
        if self.preserve_node:
            num_nodes = data.num_nodes
            num_edges = data.num_edges
            edge_index = data.edge_index

            # Indices for the nodes representing "edges"
            new_node_indices = torch.arange(num_nodes, num_nodes + num_edges, device=edge_index.device)
            # Insert the "edge" nodes into the edges
            data.edge_index = torch.cat([
                torch.stack([edge_index[0], new_node_indices], dim=0),
                torch.stack([new_node_indices, edge_index[1]], dim=0),
            ], dim=1)

            if data.edge_attr is not None:
                node_attr = data.x
                edge_attr = data.edge_attr

                # Merge edge attributes to nodes
                data.x = torch.cat([node_attr, edge_attr], dim=0)

                del data.edge_attr
            else:
                # The num_nodes must be manually set since data.x isn't updated
                data.num_nodes = num_nodes + num_edges
            return data
        raise NotImplementedError


class TokenEmbedding(Module):
    r"""
    Hold separate embeddings for different types of tokens
    and adds type embeddings to the tokens.
    """

    def __init__(self, embed_dim, embed_value: List[Union[int, torch.nn.Embedding, Module]]):
        r"""
        :param embed_dim: The number of features for each node
        :param embed_value: A list containing num_nodes, the embedding dict, or mixed, for each type.
        """
        super(TokenEmbedding, self).__init__()
        self.embed_token = []
        for i, item in enumerate(embed_value):
            if isinstance(item, int):
                # TODO: should we use sparse gradient?
                # Keep in mind that only a limited number of optimizers support sparse gradients:
                # currently it’s optim.SGD(CUDA and CPU), optim.SparseAdam(CUDA and CPU) and optim.Adagrad(CPU)
                # See https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html
                item = torch.nn.Embedding(item, embed_dim)
            elif isinstance(item, torch.nn.Embedding):
                assert item.embedding_dim == embed_dim
            self.add_module(f'embed_token_{i}', item)
            self.embed_token.append(item)

        self.embed_type = torch.nn.Embedding(len(embed_value), embed_dim)

    def forward(self, node_type, node_id) -> torch.FloatTensor:
        # Node type embedding as a base
        feat = self.embed_type(node_type)
        for i, embed in enumerate(self.embed_token):
            mask = node_type == i
            # Add token embedding
            # TODO: check whether the in-place operation works with gradients
            feat[mask] += embed(node_id[mask])
        return feat


class LabelSmoothingLoss(torch.nn.Module):
    def __init__(self, smoothing: float = 0.1,
                 reduction="mean", weight=None):
        super(LabelSmoothingLoss, self).__init__()
        self.smoothing = smoothing
        self.reduction = reduction
        self.weight = weight

    def reduce_loss(self, loss):
        return loss.mean() if self.reduction == 'mean' else loss.sum() \
            if self.reduction == 'sum' else loss

    def linear_combination(self, x, y):
        return self.smoothing * x + (1 - self.smoothing) * y

    def forward(self, preds, target):
        assert 0 <= self.smoothing < 1

        if self.weight is not None:
            self.weight = self.weight.to(preds.device)

        n = preds.size(-1)
        log_preds = F.log_softmax(preds, dim=-1)
        loss = self.reduce_loss(-log_preds.sum(dim=-1))
        nll = F.nll_loss(
            log_preds, target, reduction=self.reduction, weight=self.weight
        )
        return self.linear_combination(loss / n, nll)


class KGTransformer(Module):
    r"""
    This transformer takes in a subgraph of the KG and output the embeddings.
    """

    class TokenType(enum.IntEnum):
        Ent = 0
        MaskEnt = 1
        Rel = 2
        MaskRel = 1

    def __init__(self, num_nodes: int, relation_cnt: int, config):
        super(KGTransformer, self).__init__()
        self.d_model = config['hidden_size']
        self.num_nodes = num_nodes
        self.relation_cnt = relation_cnt
        self.line_graph = ToTripleGraph()
        # Check all appearances of token_embed before changing the scheme!
        self.token_embed = TokenEmbedding(self.d_model, embed_value=[
            self.num_nodes,
            1,
            relation_cnt * 2,
        ])
        self.graph_transformer = GraphTransformer(
            d_model=self.d_model,
            nhead=config['num_heads'],
            num_encoder_layers=config['num_layers'],
            dim_feedforward=config['dim_feedforward'],
            activation=torch.nn.LeakyReLU(negative_slope=0.01),
            moe=config['moe'],
            moe_num_expert=config['moe_num_expert'],
            moe_top_k=config['moe_top_k'],
        )

        self.pred_ent_proj = torch.nn.Linear(self.d_model, self.num_nodes)
        self.pred_rel_proj = torch.nn.Linear(self.d_model, self.relation_cnt * 2)  # 2 for inverse relations

        self.loss_type = config['loss']
        self.smoothing = config['smoothing']

    def forward(self, data: Graph) -> GraphEmbed:
        r"""
        :param data: a subgraph of the KG passed to __init__(). See graph_utils.Graph.
        :return: a graph with data.{x, edge_attr} being the embeddings
        """
        # Attribute format: [token_type, token_id, node_role]
        # Understanding: token_type is for the input, and node_role is for the output
        x = self._add_type(data.x, self.TokenType.Ent, self.TokenType.MaskEnt, 0)
        edge_attr = self._add_type(data.edge_attr, self.TokenType.Rel, self.TokenType.MaskRel, 1)
        edge_attr_inv = self._add_type(data.edge_attr + self.relation_cnt, self.TokenType.Rel, self.TokenType.MaskRel,
                                       2)

        # Add inverse edges
        edge_index = torch.cat([
            data.edge_index,
            data.edge_index[[1, 0]]
        ], dim=1)
        edge_attr = torch.cat([edge_attr, edge_attr_inv], dim=0)

        data = Graph(x=x, edge_index=edge_index, edge_attr=edge_attr)
        data_l = self.line_graph(data)
        feat = self.token_embed(data_l.x.T[0], data_l.x.T[1])
        embed = self.graph_transformer(feat, data_l.edge_index)

        node_role = data_l.x[:, 2]

        return GraphEmbed(
            x=embed[node_role == 0],
            edge_attr=embed[node_role == 1],
            inv_edge_attr=embed[node_role == 2],
        )

    def predict(self, data: Graph) -> GraphEmbed:
        r"""
        :param data: similar to forward()
        :return: a graph with data.{x, edge_attr} being the probability(score) distribution
        """
        embed = self(data)
        pred_ent = self.pred_ent_proj(embed.x)
        pred_rel = self.pred_rel_proj(embed.edge_attr)
        inv_pred_rel = self.pred_rel_proj(embed.inv_edge_attr)
        graph = GraphEmbed(x=pred_ent, edge_index=embed.edge_index, edge_attr=pred_rel, inv_edge_attr=inv_pred_rel)
        return graph

    def answer_queries(self, data: GraphWithAnswer, pred=None):
        if pred is None:
            pred = self.predict(data)
        if min(data.union_query.shape) != 0:
            sfm = torch.nn.Softmax(dim=1)
            jx_mask = data.joint_nodes
            ux_mask = data.union_query
            jpred = sfm(pred.x[jx_mask])
            # upred = sfm(pred.x[ux_mask])
            even = filter(lambda x: x % 2 == 0, range(jpred.shape[0]))
            odd = filter(lambda x: x % 2 == 1, range(jpred.shape[0]))
            ei = torch.tensor(list(even), device=data.x.device, dtype=torch.long)
            oi = torch.tensor(list(odd), device=data.x.device, dtype=torch.long)
            joint = jpred[::2] + jpred[1::2]
            x_pred = joint
            edge_mask = GraphWithAnswer.get_edge_pred_indices(data)
        else:
            x_mask = GraphWithAnswer.get_x_pred_indices(data)
            edge_mask = GraphWithAnswer.get_edge_pred_indices(data)
            x_pred = pred.x[x_mask]
        edge_pred = pred.edge_attr[edge_mask]
        inv_edge_pred = pred.inv_edge_attr[edge_mask]
        if hasattr(data, 'x_pred_mask'):
            add_mask = data.x_pred_mask
            if add_mask.dtype == torch.long:
                assert add_mask.shape[0] == 2
                assert len(add_mask.shape) == 2
                x_pred[add_mask[0], add_mask[1]] = float('-inf')
            elif not add_mask.is_floating_point():
                x_pred[add_mask] = float('-inf')
            else:
                x_pred += add_mask
        return x_pred, edge_pred, inv_edge_pred

    def forward_loss(self, data: GraphWithAnswer):
        pred = self.predict(data)
        x_pred, edge_pred, inv_edge_pred = self.answer_queries(data, pred)
        if self.loss_type == 'LS':
            loss = LabelSmoothingLoss(smoothing=self.smoothing, reduction='sum')
        elif self.loss_type == 'CE':
            loss = torch.nn.CrossEntropyLoss(reduction='sum')
        output = loss(edge_pred, data.edge_ans)
        output += loss(inv_edge_pred, data.edge_ans + self.relation_cnt)
        output /= 2
        weight_sum = torch.tensor(edge_pred.shape[0], dtype=torch.float, device=output.device)
        if hasattr(data, 'x_pred_mask'):
            if hasattr(data, 'x_pred_weight'):
                x_loss = torch.nn.CrossEntropyLoss(reduction='none')
                output += torch.sum(x_loss(x_pred, data.x_ans) * data.x_pred_weight)
                weight_sum = weight_sum + data.x_pred_weight.sum()
            else:
                output += loss(x_pred, data.x_ans)
                # noinspection PyUnresolvedReferences
                weight_sum += x_pred.shape[0]
        else:
            from metric import loss_cross_entropy_multi_ans
            x_query = GraphWithAnswer.get_x_pred_indices(data)
            if x_query is not None:
                l, w = loss_cross_entropy_multi_ans(
                    pred.x,
                    x_query, data.x_ans,
                    x_query, data.x_ans,
                    data.x_pred_weight if hasattr(data, 'x_pred_weight') else None,
                )
                output += l
                weight_sum += w

        return output, weight_sum

    @staticmethod
    def _add_type(x: torch.LongTensor, t: int, masked_t: int, node_role: int):
        r"""
        Add type info to the feature
        :param x: ids
        :param t: type
        :param masked_t: the type for x==-1
        :return: stacked features
        """
        mask = x == -1
        type_list = torch.full(x.shape, t, dtype=torch.long, device=x.device)
        type_list[mask] = masked_t
        node_role_list = torch.full(x.shape, node_role, dtype=torch.long, device=x.device)
        x = torch.stack([type_list, x, node_role_list], dim=0)
        x[1][mask] = 0
        return x.T


class KGTransformerLoss(Module):
    def __init__(self, model: Module):
        super(KGTransformerLoss, self).__init__()
        self.model = model

    def forward(self, data):
        return self.model.forward_loss(data)


class KGTransformerPredict(Module):
    def __init__(self, model: Module):
        super(KGTransformerPredict, self).__init__()
        self.model = model

    def forward(self, data):
        return self.model.answer_queries(data)


# Code from https://github.com/microsoft/Graphormer/blob/ogb-lsc/OGB-LSC/graphormer/src/model.py


class FeedForwardNetwork(nn.Module):
    def __init__(self, hidden_size, ffn_size, dropout_rate):
        super(FeedForwardNetwork, self).__init__()

        self.layer1 = nn.Linear(hidden_size, ffn_size)
        self.gelu = nn.GELU()
        self.layer2 = nn.Linear(ffn_size, hidden_size)

    def forward(self, x):
        x = self.layer1(x)
        x = self.gelu(x)
        x = self.layer2(x)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, attention_dropout_rate, head_size):
        super(MultiHeadAttention, self).__init__()

        self.head_size = head_size

        self.att_size = att_size = hidden_size // head_size
        self.scale = att_size ** -0.5

        self.linear_q = nn.Linear(hidden_size, head_size * att_size)
        self.linear_k = nn.Linear(hidden_size, head_size * att_size)
        self.linear_v = nn.Linear(hidden_size, head_size * att_size)
        self.att_dropout = nn.Dropout(attention_dropout_rate)

        self.output_layer = nn.Linear(head_size * att_size, hidden_size)

    def forward(self, q, k, v, attn_bias=None):
        orig_q_size = q.size()

        d_k = self.att_size
        d_v = self.att_size
        batch_size = q.size(0)

        # head_i = Attention(Q(W^Q)_i, K(W^K)_i, V(W^V)_i)
        q = self.linear_q(q).view(batch_size, -1, self.head_size, d_k)
        k = self.linear_k(k).view(batch_size, -1, self.head_size, d_k)
        v = self.linear_v(v).view(batch_size, -1, self.head_size, d_v)

        q = q.transpose(1, 2)  # [b, h, q_len, d_k]
        v = v.transpose(1, 2)  # [b, h, v_len, d_v]
        k = k.transpose(1, 2).transpose(2, 3)  # [b, h, d_k, k_len]

        # Scaled Dot-Product Attention.
        # Attention(Q, K, V) = softmax((QK^T)/sqrt(d_k))V
        q = q * self.scale
        x = torch.matmul(q, k)  # [b, h, q_len, k_len]
        if attn_bias is not None:
            x = x + attn_bias

        x = torch.softmax(x, dim=3)
        x = self.att_dropout(x)
        x = x.matmul(v)  # [b, h, q_len, attn]

        x = x.transpose(1, 2).contiguous()  # [b, q_len, h, attn]
        x = x.view(batch_size, -1, self.head_size * d_v)

        x = self.output_layer(x)

        assert x.size() == orig_q_size
        return x


class EncoderLayer(nn.Module):
    def __init__(self, hidden_size, ffn_size, dropout_rate, attention_dropout_rate, head_size, moe_num_expert=32,
                 moe_top_k=2, moe=True):
        super(EncoderLayer, self).__init__()

        self.self_attention_norm = nn.LayerNorm(hidden_size)
        self.self_attention = MultiHeadAttention(hidden_size, attention_dropout_rate, head_size)
        self.self_attention_dropout = nn.Dropout(dropout_rate)

        self.moe = moe
        if self.moe:
            assert ffn_size % moe_top_k == 0
            self.moeffn = CustomizedMoEPositionwiseFF(hidden_size, ffn_size // moe_top_k, dropout_rate,
                                                      True, False, moe_num_expert, moe_top_k)
        else:
            self.ffn = FeedForwardNetwork(hidden_size, ffn_size, dropout_rate)
            self.ffn_norm = nn.LayerNorm(hidden_size)
            self.ffn_dropout = nn.Dropout(dropout_rate)

    def forward(self, x, attn_bias=None):
        y = self.self_attention_norm(x)
        y = self.self_attention(y, y, y, attn_bias)
        y = self.self_attention_dropout(y)
        x = x + y

        if self.moe:
            x = self.moeffn(x)
        else:
            y = self.ffn_norm(x)
            y = self.ffn(y)
            y = self.ffn_dropout(y)
            x = x + y
        return x


# noinspection SpellCheckingInspection
class D_KGTransformer(Module):
    r"""
    A re-implementation of KGTransformer that uses deterministic operations
    """

    class TokenType(enum.IntEnum):
        Ent = 0
        MaskEnt = 1
        Rel = 2
        MaskRel = 1

    def __init__(self, num_nodes: int, relation_cnt: int, config):
        super(D_KGTransformer, self).__init__()
        self.d_model = config['hidden_size']
        self.num_nodes = num_nodes
        self.num_heads = config['num_heads']
        self.relation_cnt = relation_cnt
        # Check all appearances of token_embed before changing the scheme!
        self.token_embed = TokenEmbedding(self.d_model, embed_value=[
            self.num_nodes,
            1,
            relation_cnt * 2,
        ])
        self.attn_bias_embed = nn.Embedding(40, self.num_heads, padding_idx=1)
        with torch.no_grad():
            self.attn_bias_embed.weight[1] = torch.full((self.num_heads,), float('-inf'))
        self.encode_layers = nn.ModuleList([
            EncoderLayer(
                hidden_size=config['hidden_size'],
                ffn_size=config['dim_feedforward'],
                dropout_rate=config['dropout'],
                attention_dropout_rate=config['attention_dropout'],
                head_size=config['num_heads'],
                moe_num_expert=config['moe_num_expert'],
                moe_top_k=config['moe_top_k']
            )
            for _ in range(config['num_layers'])
        ])
        self.final_ln = nn.LayerNorm(self.d_model)

        self.pred_ent_proj = torch.nn.Linear(self.d_model, self.num_nodes)

        self.loss_type = config['loss']
        self.smoothing = config['smoothing']

    def forward(self, data: BatchMatGraph):
        feat = self.token_embed(data.embed_type, data.x)
        feat = feat.view(data.num_graphs, data.num_nodes_per_graph, self.d_model)
        rel_pos_bias = self.attn_bias_embed(data.attn_bias_type)
        # [n_graph, n_node, n_node, n_head] -> [n_graph, n_head, n_node, n_node]
        rel_pos_bias = rel_pos_bias.permute(0, 3, 1, 2)
        attn_bias = rel_pos_bias
        for layer in self.encode_layers:
            feat = layer(feat, attn_bias)
        feat = self.final_ln(feat)
        feat = feat.view(-1, self.d_model)
        return feat

    def answer_queries(self, data: BatchMatGraph):
        r"""
        :param data: BatchMatGraph
        :return:
        """
        feat = self(data)
        device = data.x.device
        relabel_arr = torch.empty(data.x.shape, dtype=torch.long, device=device)
        # Currently supports query type 0 (entities) only
        mask = data.pred_type == 0
        mask_cnt = torch.count_nonzero(mask).item()

        # relabel all the nodes
        relabel_arr[mask] = torch.arange(mask_cnt, device=device)

        if min(data.joint_nodes.shape) != 0:
            sfm = torch.nn.Softmax(dim=1)
            q_mask = mask[data.x_query]
            jq_mask = mask[data.joint_nodes]
            uq_mask = mask[data.union_query]
            p_mask = mask[data.pos_x]

            x_pred = self.pred_ent_proj(feat[mask])
            x_pred = x_pred.double()
            jq = data.joint_nodes[jq_mask]
            uq = data.union_query[uq_mask]
            assert sum(jq) == sum(data.joint_nodes)
            assert sum(uq) == sum(data.union_query)

            relabeled_jq_even = relabel_arr[jq[::2]]
            relabeled_jq_odd = relabel_arr[jq[1::2]]
            relabeled_uq = relabel_arr[uq]

            x_pred[relabeled_jq_even] = sfm(x_pred[relabeled_jq_even])
            x_pred[relabeled_jq_odd] = sfm(x_pred[relabeled_jq_odd])
            q_score = None
            if data.x_ans is not None:
                # q_score = torch.max(x_pred[relabeled_jq_even, data.x_ans], x_pred[relabeled_jq_odd, data.x_ans])
                e_score = x_pred[relabeled_jq_even, data.x_ans]
                o_score = x_pred[relabeled_jq_odd, data.x_ans]

            # Mask out all positive answers (including the predicted one)
            x_pred[relabel_arr[data.pos_x[p_mask]], data.pos_ans[p_mask]] = float('-inf')
            if data.x_ans is not None:
                x_pred[relabeled_jq_even, data.x_ans] = e_score
                x_pred[relabeled_jq_odd, data.x_ans] = o_score

            # Using rank as score
            eind = torch.argsort(x_pred[relabeled_jq_even], dim=1)
            fi = torch.arange(x_pred[relabeled_jq_even].shape[1], dtype=x_pred.dtype, device=x_pred.device).repeat(
                x_pred[relabeled_jq_even].shape[0], 1)
            x_pred[relabeled_jq_even] = torch.scatter(x_pred[relabeled_jq_even], 1, eind, fi)

            oind = torch.argsort(x_pred[relabeled_jq_odd], dim=1)
            fi2 = torch.arange(x_pred[relabeled_jq_odd].shape[1], dtype=x_pred.dtype, device=x_pred.device).repeat(
                x_pred[relabeled_jq_odd].shape[0], 1)
            x_pred[relabeled_jq_odd] = torch.scatter(x_pred[relabeled_jq_odd], 1, oind, fi2)

            q_pred = torch.max(x_pred[relabeled_jq_even], x_pred[relabeled_jq_odd])
            e_pred = x_pred[relabeled_jq_even]
            o_pred = x_pred[relabeled_jq_odd]
        else:
            # q_mask and p_mask: queries on entities (should all be True)
            q_mask = mask[data.x_query]
            p_mask = mask[data.pos_x]

            # predict for all the nodes
            x_pred = self.pred_ent_proj(feat[mask])
            # relabel the query
            relabeled_query = relabel_arr[data.x_query[q_mask]]

            # If we are training, we have to make sure that answers are not masked
            q_score = None
            if data.x_ans is not None:
                q_score = x_pred[relabeled_query, data.x_ans[q_mask]]

            # Mask out all positive answers (including the predicted one)
            x_pred[relabel_arr[data.pos_x[p_mask]], data.pos_ans[p_mask]] = float('-inf')
            q_pred = x_pred[relabeled_query]

            # Add back those to be predicted so that we know the scores of the x_ans
            if q_score is not None:
                q_pred[torch.arange(q_mask.shape[0], device=device), data.x_ans[q_mask]] = q_score
        return q_pred, None, None

    def forward_loss(self, data: BatchMatGraph):
        feat = self(data)
        device = data.x.device
        relabel_arr = torch.empty(data.x.shape, dtype=torch.long, device=device)
        # Currently supports query type 0 (entities) only
        mask = data.pred_type == 0
        mask_cnt = torch.count_nonzero(mask).item()

        # relable all the nodes
        relabel_arr[mask] = torch.arange(mask_cnt, device=device)

        from metric import loss_cross_entropy_multi_ans, loss_label_smoothing_multi_ans
        q_mask = mask[data.x_query]
        p_mask = mask[data.pos_x]
        if self.loss_type == "CE":
            f = feat[mask]
            l, w = loss_cross_entropy_multi_ans(
                self.pred_ent_proj(f).double(),
                relabel_arr[data.x_query[q_mask]], data.x_ans[q_mask],
                relabel_arr[data.pos_x[p_mask]], data.pos_ans[p_mask],
                query_w=data.x_pred_weight[q_mask],
            )
        elif self.loss_type == 'LS':
            f = feat[mask]
            l, w = loss_label_smoothing_multi_ans(
                self.pred_ent_proj(f).double(),
                relabel_arr[data.x_query[q_mask]], data.x_ans[q_mask],
                relabel_arr[data.pos_x[p_mask]], data.pos_ans[p_mask],
                self.smoothing,
                query_w=data.x_pred_weight[q_mask]
            )
        import math
        assert not math.isnan(l.item())
        return l, w


# noinspection SpellCheckingInspection
class D_KGTransformerLoss(Module):
    def __init__(self, model: Module):
        super(D_KGTransformerLoss, self).__init__()
        self.model = model

    def forward(self, data):
        return self.model.forward_loss(data)
