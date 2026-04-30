from audioop import bias
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter, scatter_mean

from .utils import scatter_batch, get_intergraph_edge_index


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2):
        super().__init__()
        assert num_layers >= 2
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.SiLU())
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.SiLU())
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


class TransformerBlock(nn.Module):
    def __init__(self, dim, nhead):
        super().__init__()
        self.attn_norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, nhead, batch_first=True)
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = MLP(dim, dim*4, dim)

    def forward(self, x, key_padding_mask=None):
        x_norm = self.attn_norm(x)
        x = x + self.attn(
            x_norm, x_norm, x_norm,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )[0]
        x = x + self.ffn(self.ffn_norm(x))
        return x


class TokenInteractionModule(nn.Module):
    def __init__(
        self,
        dim,
        num_layer=4,
        num_heads=8,
    ):
        super().__init__()
        self.blocks = nn.ModuleList([TransformerBlock(dim, num_heads) for _ in range(num_layer)])

    def forward(self, x, key_padding_mask=None):
        for block in self.blocks:
            x = block(x, key_padding_mask=key_padding_mask)
        return x


# ================== GNN ==================
class GraphEncoderBlock(nn.Module):
    def __init__(self, edge_attr_dim, hidden_dim):
        super().__init__()
        self.mlp_msg = MLP(2 * hidden_dim + edge_attr_dim, hidden_dim, hidden_dim)
        self.mlp_node = MLP(2 * hidden_dim, hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, h, edge_index, edge_attr):
        row, col = edge_index
        if edge_attr.dim() == 1:
            edge_attr = edge_attr.unsqueeze(-1)
        msg = torch.cat([h[row], h[col], edge_attr], dim=-1)
        msg = self.mlp_msg(msg)
        msg_agg = scatter(msg, row, dim=0, dim_size=h.size(0), reduce='mean')
        h_new = self.mlp_node(torch.cat([h, msg_agg], dim=-1))
        h = self.norm(h + h_new)
        return h


class GraphEncoder(nn.Module):
    def __init__(self, num_layer, node_input_dim, edge_attr_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Linear(node_input_dim, hidden_dim)
        self.layers = nn.ModuleList(
            [GraphEncoderBlock(edge_attr_dim, hidden_dim)for _ in range(num_layer)])

    def forward(self, h, edge_index, edge_attr):
        h = self.embedding(h)
        for layer in self.layers:
            h = layer(h, edge_index, edge_attr)
        return h


class AtomInteractionGraphBlock(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.mlp_msg = MLP(2 * hidden_dim, hidden_dim, hidden_dim)
        self.mlp_upd = MLP(2 * hidden_dim, hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, h, inter_edge_index):
        row, col = inter_edge_index
        msg = torch.cat([h[row], h[col]], dim=-1)
        msg = self.mlp_msg(msg)
        agg = scatter(msg, row, dim=0, dim_size=h.size(0), reduce='mean')
        h_new = self.mlp_upd(torch.cat([h, agg], dim=-1))
        h = self.norm(h + h_new)
        return h


class AtomInteractionModule(nn.Module):
    def __init__(self, hidden_dim, num_layer):
        super().__init__()
        self.layers = nn.ModuleList([AtomInteractionGraphBlock(hidden_dim) for _ in range(num_layer)])

    def forward(self, h, inter_edge_index):
        for layer in self.layers:
            h = layer(h, inter_edge_index)
        return h


class AtomToTokenPool(nn.Module):
    def __init__(self, hidden_dim, num_layer):
        super().__init__()
        if num_layer > 0:
            self.layers = nn.ModuleList([AtomInteractionGraphBlock(hidden_dim) for _ in range(num_layer)])
        else:
            self.layers = None

    def forward(self, h_atom, inter_edge_index, n2g_indices):
        if self.layers is not None:
            for layer in self.layers:
                h_atom = layer(h_atom, inter_edge_index)

        h_graph = scatter_mean(h_atom, n2g_indices, dim=0)
        return h_graph


# ================== Main Model ==================
class BaseMLP_2(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        num_layer=4,
        activation=None,
        residual=False,
        last_act=False,
    ):
        super().__init__()
        activation = nn.SiLU() if activation is None else activation
        self.residual = residual
        self.num_layer = num_layer
        if residual:
            assert output_dim == input_dim

        def make_block(in_dim, out_dim, use_norm=True, use_act=True):
            layers = [nn.Linear(in_dim, out_dim)]
            if use_norm:
                layers.append(nn.LayerNorm(out_dim))
            if use_act:
                layers.append(activation)
            return nn.Sequential(*layers)

        self.mlps = nn.ModuleList()
        self.mlps.append(make_block(input_dim, hidden_dim,
                         use_norm=True, use_act=True))
        for _ in range(self.num_layer - 2):
            self.mlps.append(make_block(
                hidden_dim, hidden_dim, use_norm=True, use_act=True))
        self.mlps.append(
            make_block(hidden_dim, output_dim, use_norm=(
                not last_act), use_act=last_act)
        )
        if self.residual:
            self.res_scale = nn.Parameter(torch.zeros(len(self.mlps)))

    def forward(self, x):
        if self.residual:
            for i, mlp in enumerate(self.mlps):
                x = x + self.res_scale[i] * mlp(x)
        else:
            for mlp in self.mlps:
                x = mlp(x)
        return x


class SolvMix(nn.Module):
    def __init__(
        self,
        num_gnn_blocks=3,
        node_input_dim=152,
        edge_attr_dim=13,
        hidden_dim=64,
        num_mlp_layer=3,
        num_token_blocks=3,
        num_atten_head=4,
        num_atom_blocks=0,
        seq_len=16,
        device='cpu',
        use_amount_scale=False,
        use_amount_and_type_emb=False,
        use_res_scale=False,
        cond_mode="late_concat",
    ):
        super(SolvMix, self).__init__()
        self.seq_len = seq_len
        self.use_amount_scale = use_amount_scale
        self.use_amount_and_type_emb = use_amount_and_type_emb
        self.cond_mode = cond_mode

        self.solvent_encoder = GraphEncoder(
            num_layer=num_gnn_blocks,
            node_input_dim=node_input_dim,
            edge_attr_dim=edge_attr_dim,
            hidden_dim=hidden_dim,
        )
        self.salt_encoder = GraphEncoder(
            num_layer=num_gnn_blocks,
            node_input_dim=node_input_dim,
            edge_attr_dim=edge_attr_dim,
            hidden_dim=hidden_dim,
        )

        self.atom_to_token_pool = AtomToTokenPool(hidden_dim, num_atom_blocks)

        self.token_interaction_module = TokenInteractionModule(
            hidden_dim,
            num_layer=num_token_blocks,
            num_heads=num_atten_head,
        )

        self.pool_proj = nn.Linear(hidden_dim, hidden_dim)

        self.temp_token_mlp = MLP(2, hidden_dim, hidden_dim)
        self.conc_token_mlp = MLP(2, hidden_dim, hidden_dim)

        if self.use_amount_and_type_emb:
            self.type_emb = nn.Embedding(2, hidden_dim)
            self.solvent_amount_emb = MLP(1, hidden_dim, hidden_dim)
            self.salt_amount_emb = MLP(1, hidden_dim, hidden_dim)

        if self.use_amount_scale:
            self.solvent_gate_mlp = MLP(1, hidden_dim, hidden_dim)
            self.salt_gate_mlp = MLP(1, hidden_dim, hidden_dim)

        # New SOTA
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim+2, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
            MLP(hidden_dim, hidden_dim, hidden_dim, num_layers=3),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
            MLP(hidden_dim, hidden_dim+2, 1, num_layers=4),
            nn.Softplus(),
        )

        # Static structure cache
        self._cache_enabled = False
        self._split_caches = {}

    def enable_cache(self):
        self._cache_enabled = True

    def clear_cache(self):
        self._cache_enabled = False
        self._split_caches.clear()

    def _make_cache_key(self, n2g_indices, n2b_indices):
        return (n2g_indices.shape, n2b_indices.shape)

    def forward(
        self,
        batch,
        salt_batch,
        n2g_indices,
        n2b_indices,
        g2b_indices,
        T,
        c,
        ratios=None,
        pos_idx=None,
        seq_len=None,
        num_solvent_graphs=None,
    ):
        if self._cache_enabled:
            key = self._make_cache_key(n2g_indices, n2b_indices)
            if key not in self._split_caches:
                self._split_caches[key] = {}
            cache = self._split_caches[key]
        else:
            cache = None

        h_atom_solv = self.solvent_encoder(batch.x, batch.edge_index, batch.edge_attr)
        h_atom_salt = self.salt_encoder(salt_batch.x, salt_batch.edge_index, salt_batch.edge_attr)
        h_atom = torch.cat([h_atom_solv, h_atom_salt], dim=0)

        if cache is not None and 'inter_edge_index' in cache:
            inter_edge_index = cache['inter_edge_index']
        else:
            inter_edge_index = get_intergraph_edge_index(n2g_indices, n2b_indices)
            if cache is not None:
                cache['inter_edge_index'] = inter_edge_index

        h_graph = self.atom_to_token_pool(h_atom, inter_edge_index, n2g_indices)

        if self.use_amount_and_type_emb:
            _n_solv = num_solvent_graphs if num_solvent_graphs is not None else (
                ratios.shape[0] if ratios is not None else 0)
            # type embedding
            token_types = torch.zeros(h_graph.size(0), dtype=torch.long, device=h_graph.device)
            if _n_solv < h_graph.size(0):
                token_types[_n_solv:] = 1
            h_graph = h_graph + self.type_emb(token_types)
            # solvent amount embedding
            if _n_solv > 0:
                solvent_amount = self.solvent_amount_emb(ratios.unsqueeze(-1))
                h_graph[:_n_solv] = h_graph[:_n_solv] + solvent_amount
            # salt amount embedding
            _n_salt = h_graph.size(0) - _n_solv
            if _n_salt > 0:
                salt_g2b = g2b_indices[_n_solv:]
                salt_c = c[salt_g2b].unsqueeze(-1)
                salt_amount = self.salt_amount_emb(salt_c)
                h_graph[_n_solv:] = h_graph[_n_solv:] + salt_amount

        if self.use_amount_scale:
            if cache is not None and 'num_solvent_graphs' in cache:
                _n_solv = cache['num_solvent_graphs']
            else:
                _n_solv = num_solvent_graphs if num_solvent_graphs is not None else (
                    ratios.shape[0] if ratios is not None else 0)
                if cache is not None:
                    cache['num_solvent_graphs'] = _n_solv

            gated_parts = []
            if _n_solv > 0:
                if cache is not None and 'solvent_ratios_for_gate' in cache:
                    solvent_ratios_for_gate = cache['solvent_ratios_for_gate']
                else:
                    solvent_ratios_for_gate = ratios.unsqueeze(-1)
                    if cache is not None:
                        cache['solvent_ratios_for_gate'] = solvent_ratios_for_gate
                solvent_gate = self.solvent_gate_mlp(solvent_ratios_for_gate)
                gated_parts.append(h_graph[:_n_solv] * solvent_gate)

            if cache is not None and 'salt_c_for_gate' in cache:
                salt_c_for_gate = cache['salt_c_for_gate']
            else:
                salt_c_for_gate = c.unsqueeze(-1)
                if cache is not None:
                    cache['salt_c_for_gate'] = salt_c_for_gate
            salt_gate = self.salt_gate_mlp(salt_c_for_gate)
            gated_parts.append(h_graph[_n_solv:] * salt_gate)

            h_graph = torch.cat(gated_parts, dim=0)

        batch_size = int(n2b_indices.max().item()) + 1
        actual_seq_len = seq_len if seq_len is not None else self.seq_len

        if cache is not None and 'scatter_indices' in cache:
            device = h_graph.device
            hidden_dim = h_graph.size(-1)
            h_graph_scattered = torch.zeros(
                batch_size, actual_seq_len, hidden_dim, device=device)
            h_graph_scattered_flat = h_graph_scattered.view(
                batch_size * actual_seq_len, hidden_dim)
            h_graph_scattered_flat.index_add_(
                0, cache['scatter_indices'], h_graph)
        else:
            h_graph_scattered = scatter_batch(
                h_graph, g2b_indices, pos_idx, batch_size, actual_seq_len)
            if cache is not None:
                cache['scatter_indices'] = (
                    g2b_indices * actual_seq_len + pos_idx)

        padding_mask = (h_graph_scattered.abs().sum(dim=-1) == 0)
        actual_seq_len = h_graph_scattered.size(1)

        token_input = h_graph_scattered
        token_padding_mask = padding_mask

        if self.cond_mode == "in_context":
            temp_feat = torch.stack([
                T / 273.15,
                273.15 / T.clamp_min(1e-8),
            ], dim=-1)
            conc_feat = torch.stack([
                c,
                torch.log1p(c.clamp_min(0)),
            ], dim=-1)

            temp_token = self.temp_token_mlp(temp_feat).unsqueeze(1)
            conc_token = self.conc_token_mlp(conc_feat).unsqueeze(1)

            cond_tokens = torch.cat([temp_token, conc_token], dim=1)
            cond_mask = torch.zeros(
                token_input.size(0),
                2,
                dtype=torch.bool,
                device=token_input.device,
            )

            token_input = torch.cat([token_input, cond_tokens], dim=1)
            token_padding_mask = torch.cat([token_padding_mask, cond_mask], dim=1)
        elif self.cond_mode != "late_concat":
            raise ValueError(f"Unknown cond_mode={self.cond_mode}")

        h_all = self.token_interaction_module(token_input, key_padding_mask=token_padding_mask)

        h_component = h_all[:, :actual_seq_len, :]
        h_component = h_component.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        valid_len = (~padding_mask).sum(dim=1, keepdim=True)
        h_sample = h_component.sum(dim=1) / valid_len.clamp(min=1)

        feature_all = torch.cat([self.pool_proj(h_sample), T.unsqueeze(1) / 273.15,
                                (273.15 / T.clamp_min(1e-8)).unsqueeze(1)], dim=1)

        res = self.readout(feature_all).squeeze(1)

        return res
