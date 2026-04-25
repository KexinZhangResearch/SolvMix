import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter, scatter_mean

# ================== GPU-native metrics ==================


def pearson_correlation(x, y):
    """x, y: 1-D tensors, typically on GPU."""
    x = x.float()
    y = y.float()
    x_mean = x.mean()
    y_mean = y.mean()
    x_std = x.std(unbiased=False)
    y_std = y.std(unbiased=False)
    cov = ((x - x_mean) * (y - y_mean)).mean()
    return cov / (x_std * y_std + 1e-8)


def spearman_correlation(x, y):
    """Approximate Spearman rho with GPU-native double argsort ranks."""
    x_rank = torch.argsort(torch.argsort(x)).float()
    y_rank = torch.argsort(torch.argsort(y)).float()
    return pearson_correlation(x_rank, y_rank)


def mse_metric(pred, target):
    return F.mse_loss(pred.float(), target.float())


def mae_metric(pred, target):
    return F.l1_loss(pred.float(), target.float())


def rmse_metric(pred, target):
    return torch.sqrt(mse_metric(pred, target) + 1e-12)


def r2_score(pred, target):
    pred = pred.float()
    target = target.float()
    ss_res = torch.sum((target - pred) ** 2)
    ss_tot = torch.sum((target - target.mean()) ** 2)
    return 1.0 - ss_res / (ss_tot + 1e-12)


def regression_metrics(pred, target):
    """Raw-space regression metrics for validation/test."""
    return {
        "r2": r2_score(pred, target),
        "mae": mae_metric(pred, target),
        "mse": mse_metric(pred, target),
        "rmse": rmse_metric(pred, target),
        "spearman": spearman_correlation(pred, target),
        "pearson": pearson_correlation(pred, target),
    }


def make_regression_loss(name):
    """Training loss in log-target space. name in {mse, mae, rmse}."""
    name = (name or "mse").lower()
    if name == "mse":
        return F.mse_loss
    if name == "mae":
        return F.l1_loss
    if name == "rmse":
        return lambda x, y: torch.sqrt(F.mse_loss(x, y) + 1e-12)
    raise ValueError(f"Unknown loss_type={name}; choose from mse/mae/rmse")


# ================== batching utilities ==================


def scatter_batch_with_mask(feature, sample_indices, batch_size, seq_len=16):
    """
    Robust graph-token batching.

    feature: [num_graphs, hidden_dim]
    sample_indices: [num_graphs], does not need to be sorted.
    """
    device = feature.device
    hidden_dim = feature.size(-1)
    n = sample_indices.numel()
    out = torch.zeros(batch_size, seq_len, hidden_dim, device=device)
    mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)
    if n == 0:
        return out, mask, torch.empty(0, dtype=torch.long, device=device)

    sample_indices = sample_indices.long()
    try:
        order = torch.argsort(sample_indices, stable=True)
    except TypeError:
        eps = torch.arange(n, device=device, dtype=torch.long)
        order = torch.argsort(sample_indices * (n + 1) + eps)

    sorted_samples = sample_indices[order]
    sorted_feature = feature[order]
    is_new_group = torch.cat(
        [torch.ones(1, dtype=torch.bool, device=device),
         sorted_samples[1:] != sorted_samples[:-1]], dim=0
    )
    group_id = torch.cumsum(is_new_group.long(), dim=0) - 1
    num_groups = int(group_id.max().item()) + 1

    ones = torch.ones(n, device=device, dtype=torch.float32)
    group_sizes = torch.zeros(num_groups, device=device).index_add_(0, group_id, ones)
    group_offsets = torch.cat(
        [torch.zeros(1, device=device), torch.cumsum(group_sizes[:-1], dim=0)], dim=0
    ).long()

    arange = torch.arange(n, device=device, dtype=torch.long)
    pos_sorted = arange - group_offsets[group_id]
    pos_sorted_clamped = pos_sorted.clamp(max=seq_len - 1)
    flat_indices = sorted_samples * seq_len + pos_sorted_clamped

    out.view(batch_size * seq_len, hidden_dim).index_add_(0, flat_indices, sorted_feature)
    mask.view(batch_size * seq_len).index_fill_(0, flat_indices, True)

    pos_idx = torch.empty_like(pos_sorted_clamped)
    pos_idx[order] = pos_sorted_clamped
    return out, mask, pos_idx


def scatter_scalar_batch_with_mask(values, sample_indices, batch_size, seq_len=16):
    """
    values: [num_graphs] or [num_graphs, 1]
    returns scalar tensor [B, seq_len] and mask [B, seq_len].
    """
    if values.dim() == 1:
        values = values.unsqueeze(-1)
    out, mask, pos = scatter_batch_with_mask(values, sample_indices, batch_size, seq_len)
    return out.squeeze(-1), mask, pos


def masked_mean(x, mask, dim=1, eps=1e-8):
    mask_f = mask.float().unsqueeze(-1)
    return (x * mask_f).sum(dim=dim) / (mask_f.sum(dim=dim).clamp_min(eps))


# ================== modern blocks ==================


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps) * self.weight


def make_norm(dim, norm_type="layernorm"):
    norm_type = (norm_type or "layernorm").lower()
    if norm_type == "rmsnorm":
        return RMSNorm(dim)
    if norm_type == "identity":
        return nn.Identity()
    return nn.LayerNorm(dim)


class SwiGLUFeedForward(nn.Module):
    def __init__(self, dim, mult=4.0, dropout=0.0):
        super().__init__()
        hidden = int(dim * mult)
        self.w1 = nn.Linear(dim, hidden)
        self.w2 = nn.Linear(dim, hidden)
        self.w3 = nn.Linear(hidden, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = F.silu(self.w1(x)) * self.w2(x)
        x = self.dropout(x)
        return self.w3(x)


class ModernMLP(nn.Module):
    def __init__(
        self, input_dim, hidden_dim, output_dim, num_layers=2,
        dropout=0.0, norm_type="layernorm", use_swiglu=False,
        residual=False, last_act=False
    ):
        super().__init__()
        self.residual = residual and input_dim == output_dim
        layers = []
        dim = input_dim
        for _ in range(max(num_layers - 1, 1)):
            if use_swiglu:
                layers.append(nn.Linear(dim, hidden_dim))
                layers.append(make_norm(hidden_dim, norm_type))
                layers.append(nn.SiLU())
                layers.append(nn.Dropout(dropout))
            else:
                layers.append(nn.Linear(dim, hidden_dim))
                layers.append(make_norm(hidden_dim, norm_type))
                layers.append(nn.SiLU())
                layers.append(nn.Dropout(dropout))
            dim = hidden_dim
        layers.append(nn.Linear(dim, output_dim))
        if last_act:
            layers.append(nn.SiLU())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        y = self.net(x)
        return x + y if self.residual else y


class AdaLNZeroModulator(nn.Module):
    """Condition -> shift/scale/gate for AdaLN-Zero style residual blocks."""

    def __init__(self, cond_dim, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cond_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, 6 * dim),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, cond):
        return self.net(cond).chunk(6, dim=-1)


class ModernTransformerBlock(nn.Module):
    """
    PreNorm Transformer block with SwiGLU FFN and proper padding mask.
    """

    def __init__(
        self, dim, num_heads, ffn_mult=4.0, attn_dropout=0.0,
        resid_dropout=0.0, ffn_dropout=0.0, norm_type="layernorm",
        return_attn=False, block_cond_mode="none", cond_dim=4
    ):
        super().__init__()
        self.return_attn = return_attn
        self.block_cond_mode = block_cond_mode
        self.norm1 = make_norm(dim, norm_type)
        self.attn = nn.MultiheadAttention(
            dim, num_heads, dropout=attn_dropout, batch_first=True
        )
        self.drop1 = nn.Dropout(resid_dropout)
        self.norm2 = make_norm(dim, norm_type)
        self.ffn = SwiGLUFeedForward(dim, mult=ffn_mult, dropout=ffn_dropout)
        self.drop2 = nn.Dropout(resid_dropout)
        self.modulator = (
            AdaLNZeroModulator(cond_dim=cond_dim, dim=dim)
            if block_cond_mode == "adaln_zero" else None
        )

    @staticmethod
    def _shift_scale(x, shift, scale):
        return x * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)

    def forward(self, x, mask=None, cond=None):
        key_padding_mask = None if mask is None else ~mask.bool()

        if self.modulator is not None and cond is not None:
            shift_msa, scale_msa, gate_msa, shift_ffn, scale_ffn, gate_ffn = self.modulator(cond)
            qkv = self._shift_scale(self.norm1(x), shift_msa, scale_msa)
            attn_out, attn_w = self.attn(
                qkv, qkv, qkv,
                key_padding_mask=key_padding_mask,
                need_weights=self.return_attn,
                average_attn_weights=False,
            )
            x = x + self.drop1(attn_out * gate_msa.unsqueeze(1))
            ffn_in = self._shift_scale(self.norm2(x), shift_ffn, scale_ffn)
            x = x + self.drop2(self.ffn(ffn_in) * gate_ffn.unsqueeze(1))
            return (x, attn_w) if self.return_attn else (x, None)

        x_norm = self.norm1(x)
        attn_out, attn_w = self.attn(
            x_norm, x_norm, x_norm,
            key_padding_mask=key_padding_mask,
            need_weights=self.return_attn,
            average_attn_weights=False,
        )
        x = x + self.drop1(attn_out)
        x = x + self.drop2(self.ffn(self.norm2(x)))
        return (x, attn_w) if self.return_attn else (x, None)


class TokenInteractiveModule(nn.Module):
    def __init__(
        self, dim, num_layers=3, num_heads=4, ffn_mult=4.0,
        attn_dropout=0.0, resid_dropout=0.0, ffn_dropout=0.0,
        norm_type="layernorm", return_attn=False,
        block_cond_mode="none", cond_dim=4
    ):
        super().__init__()
        self.block_cond_mode = block_cond_mode
        self.layers = nn.ModuleList([
            ModernTransformerBlock(
                dim=dim, num_heads=num_heads, ffn_mult=ffn_mult,
                attn_dropout=attn_dropout, resid_dropout=resid_dropout,
                ffn_dropout=ffn_dropout, norm_type=norm_type,
                return_attn=return_attn, block_cond_mode=block_cond_mode,
                cond_dim=cond_dim,
            )
            for _ in range(num_layers)
        ])

    def forward(self, x, mask=None, cond=None):
        attn_maps = []
        for layer in self.layers:
            x, attn = layer(x, mask=mask, cond=cond)
            if attn is not None:
                attn_maps.append(attn)
        return x, attn_maps


class MaskedAttentionPooling(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.score = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, 1)
        )

    def forward(self, x, mask):
        logits = self.score(x).squeeze(-1)
        logits = logits.masked_fill(~mask.bool(), torch.finfo(logits.dtype).min)
        weights = torch.softmax(logits, dim=1)
        pooled = torch.sum(x * weights.unsqueeze(-1), dim=1)
        return pooled, weights


# ================== graph encoders ==================


class ConditionedGraphLayer(nn.Module):
    """
    Residual GNN layer with optional atom-level condition modulation.
    """

    def __init__(
        self, edge_attr_dim, hidden_dim, cond_dim=5, dropout=0.0,
        norm_type="layernorm", atom_mod_mode="none"
    ):
        super().__init__()
        self.atom_mod_mode = atom_mod_mode
        self.cond_dim = cond_dim
        self.norm = make_norm(hidden_dim, norm_type)

        msg_in_dim = 2 * hidden_dim + edge_attr_dim
        if atom_mod_mode == "message":
            msg_in_dim += cond_dim

        self.msg = nn.Sequential(
            nn.Linear(msg_in_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.upd = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.cond_gate = None
        if atom_mod_mode == "gating":
            self.cond_gate = nn.Sequential(
                nn.Linear(cond_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Sigmoid(),
            )

        self.cond_norm = None
        if atom_mod_mode == "cond_norm":
            self.cond_norm = nn.Sequential(
                nn.Linear(cond_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, 2 * hidden_dim),
            )
            nn.init.zeros_(self.cond_norm[-1].weight)
            nn.init.zeros_(self.cond_norm[-1].bias)

        self.dropout = nn.Dropout(dropout)

    def _conditioned_norm(self, h, cond_node):
        hn = self.norm(h)
        if self.cond_norm is not None and cond_node is not None:
            shift, scale = self.cond_norm(cond_node).chunk(2, dim=-1)
            hn = hn * (1.0 + scale) + shift
        return hn

    def forward(self, h, edge_index, edge_attr, cond_node=None):
        row, col = edge_index
        if edge_attr.dim() == 1:
            edge_attr = edge_attr.unsqueeze(-1)

        h0 = h
        hn = self._conditioned_norm(h, cond_node)

        parts = [hn[row], hn[col], edge_attr]
        if self.atom_mod_mode == "message" and cond_node is not None:
            parts.append(cond_node[row])

        msg = self.msg(torch.cat(parts, dim=-1))
        if self.cond_gate is not None and cond_node is not None:
            msg = msg * self.cond_gate(cond_node[row])

        agg = scatter(msg, row, dim=0, dim_size=h.size(0), reduce="mean")
        dh = self.upd(torch.cat([hn, agg], dim=-1))

        if self.cond_gate is not None and cond_node is not None:
            dh = dh * self.cond_gate(cond_node)

        return h0 + self.dropout(dh)


class ConditionedGraphEncoder(nn.Module):
    def __init__(
        self, num_layers, node_input_dim, edge_attr_dim, hidden_dim,
        cond_dim=5, dropout=0.0, norm_type="layernorm", atom_mod_mode="none"
    ):
        super().__init__()
        self.atom_mod_mode = atom_mod_mode
        self.embedding = nn.Linear(node_input_dim, hidden_dim)
        self.layers = nn.ModuleList([
            ConditionedGraphLayer(
                edge_attr_dim=edge_attr_dim,
                hidden_dim=hidden_dim,
                cond_dim=cond_dim,
                dropout=dropout,
                norm_type=norm_type,
                atom_mod_mode=atom_mod_mode,
            )
            for _ in range(num_layers)
        ])
        self.final_norm = make_norm(hidden_dim, norm_type)

    def forward(self, x, edge_index, edge_attr, cond_node=None):
        h = self.embedding(x)
        for layer in self.layers:
            h = layer(h, edge_index, edge_attr, cond_node=cond_node)
        return self.final_norm(h)


# ================== component conditioning ==================


class FiLMConditioner(nn.Module):
    def __init__(self, dim, cond_dim=4, hidden_dim=None):
        super().__init__()
        hidden_dim = hidden_dim or dim
        self.net = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 2 * dim)
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, tokens, z):
        gamma_beta = self.net(z).unsqueeze(1)
        gamma, beta = gamma_beta.chunk(2, dim=-1)
        return tokens * (1.0 + gamma) + beta


class ConditionBuilder(nn.Module):
    """Centralizes physical/formulation condition construction."""

    SOLVENT = 0
    SALT = 1
    ADDITIVE = 2
    UNKNOWN = 3

    def __init__(self):
        super().__init__()

    @staticmethod
    def global_condition(T, c):
        return torch.stack([
            T,
            1.0 / T.clamp_min(1e-8),
            c,
            torch.log1p(c.clamp_min(0)),
        ], dim=-1)

    @staticmethod
    def build_component_metadata(
        n_graph, g2b_indices, ratios, c, dtype=None,
        graph_types=None, graph_amounts=None
    ):
        device = g2b_indices.device
        dtype = dtype or c.dtype

        if graph_types is not None and graph_amounts is not None:
            graph_types = graph_types.to(device=device, dtype=torch.long)
            graph_amounts = graph_amounts.to(device=device, dtype=dtype)
            if graph_types.numel() != n_graph:
                raise ValueError(
                    f"graph_types length mismatch: got {graph_types.numel()}, expected {n_graph}."
                )
            if graph_amounts.numel() != n_graph:
                raise ValueError(
                    f"graph_amounts length mismatch: got {graph_amounts.numel()}, expected {n_graph}."
                )
            return graph_types, graph_amounts

        num_solvent = int(ratios.numel()) if ratios is not None else 0
        num_solvent = min(num_solvent, n_graph)
        graph_types = torch.zeros(n_graph, dtype=torch.long, device=device)
        if num_solvent < n_graph:
            graph_types[num_solvent:] = ConditionBuilder.SALT

        graph_amounts = torch.zeros(n_graph, dtype=dtype, device=device)
        if ratios is not None and num_solvent > 0:
            graph_amounts[:num_solvent] = ratios[:num_solvent].to(device=device, dtype=dtype)
        if num_solvent < n_graph:
            salt_sample_idx = g2b_indices[num_solvent:]
            graph_amounts[num_solvent:] = c[salt_sample_idx].to(dtype)
        return graph_types, graph_amounts

    def atom_condition(self, T, c, n2b_indices, n2g_indices, graph_amounts):
        global_cond = self.global_condition(T, c)
        atom_sample = n2b_indices.long()
        atom_amount = graph_amounts[n2g_indices.long()].unsqueeze(-1)
        return torch.cat([global_cond[atom_sample], atom_amount], dim=-1)


class ContextTokenBuilder(nn.Module):
    """
    Build condition/context tokens for condition-as-context interaction.
    """

    SOLVENT = 0
    SALT = 1

    def __init__(
        self, dim, use_temp=True, use_comp=True, use_global=True,
        separate_amount_emb=True
    ):
        super().__init__()
        self.dim = dim
        self.use_temp = use_temp
        self.use_comp = use_comp
        self.use_global = use_global
        self.separate_amount_emb = separate_amount_emb

        if use_temp:
            self.temp_mlp = ModernMLP(
                2, dim, dim, num_layers=2, norm_type="identity")

        if use_comp:
            self.solvent_stats_mlp = ModernMLP(
                4, dim, dim, num_layers=2, norm_type="identity")
            self.solvent_amount_mlp = ModernMLP(
                1, dim, dim, num_layers=2, norm_type="identity")
            self.solvent_fuse = ModernMLP(
                2 * dim, dim, dim, num_layers=2, norm_type="identity")

            self.salt_stats_mlp = ModernMLP(
                5, dim, dim, num_layers=2, norm_type="identity")
            self.salt_amount_mlp = ModernMLP(
                1, dim, dim, num_layers=2, norm_type="identity")
            self.salt_fuse = ModernMLP(
                2 * dim, dim, dim, num_layers=2, norm_type="identity")

        if use_global:
            self.global_token = nn.Parameter(torch.zeros(1, 1, dim))
            self.global_cond = ModernMLP(
                4, dim, dim, num_layers=2, norm_type="identity")
            nn.init.normal_(self.global_token, std=0.02)

    def _safe_stats(self, amount_tensor, mask):
        mask = mask.bool()
        a = amount_tensor.masked_fill(~mask, 0.0)
        count = mask.float().sum(dim=1, keepdim=True)
        safe_count = count.clamp_min(1.0)
        sum_a = a.sum(dim=1, keepdim=True)
        max_a = a.masked_fill(~mask, torch.finfo(a.dtype).min).max(dim=1, keepdim=True).values
        max_a = torch.where(count > 0, max_a, torch.zeros_like(max_a))
        p = a / sum_a.clamp_min(1e-8)
        entropy = -(
            p.clamp_min(1e-8) * p.clamp_min(1e-8).log()
        ).masked_fill(~mask, 0.0).sum(dim=1, keepdim=True)
        return sum_a, max_a, entropy, safe_count

    def _masked_amount_pool(self, amount_tensor, mask, mlp):
        emb = mlp(amount_tensor.unsqueeze(-1))
        return masked_mean(emb, mask)

    def forward(self, component_tokens, component_mask, T, c, amount_tensor, type_tensor=None):
        B, S, D = component_tokens.shape
        device = component_tokens.device
        z = ConditionBuilder.global_condition(T, c)
        ctx_tokens, ctx_names = [], []

        if self.use_temp:
            temp_in = torch.stack([T, 1.0 / T.clamp_min(1e-8)], dim=-1)
            ctx_tokens.append(self.temp_mlp(temp_in).unsqueeze(1))
            ctx_names.append("temperature")

        if self.use_comp:
            if type_tensor is None:
                type_tensor = torch.zeros(B, S, dtype=torch.long, device=device)
            type_tensor = type_tensor.long()
            solvent_mask = component_mask.bool() & (type_tensor == self.SOLVENT)
            salt_mask = component_mask.bool() & (type_tensor == self.SALT)

            solv_sum, solv_max, solv_entropy, solv_count = self._safe_stats(
                amount_tensor, solvent_mask)
            solv_stats = torch.cat(
                [solv_sum, solv_max, solv_entropy, solv_count], dim=-1)
            solv_stats_token = self.solvent_stats_mlp(solv_stats)
            solv_amount_token = self._masked_amount_pool(
                amount_tensor, solvent_mask, self.solvent_amount_mlp)
            solvent_context = self.solvent_fuse(
                torch.cat([solv_stats_token, solv_amount_token], dim=-1))
            ctx_tokens.append(solvent_context.unsqueeze(1))
            ctx_names.append("solvent_composition")

            salt_sum, salt_max, _salt_entropy_unused, salt_count = self._safe_stats(
                amount_tensor, salt_mask)
            salt_stats = torch.cat([
                salt_sum,
                salt_max,
                salt_count,
                c.unsqueeze(-1),
                torch.log1p(c.clamp_min(0)).unsqueeze(-1),
            ], dim=-1)
            salt_stats_token = self.salt_stats_mlp(salt_stats)
            salt_amount_token = self._masked_amount_pool(
                amount_tensor, salt_mask, self.salt_amount_mlp)
            salt_context = self.salt_fuse(
                torch.cat([salt_stats_token, salt_amount_token], dim=-1))
            ctx_tokens.append(salt_context.unsqueeze(1))
            ctx_names.append("salt_concentration")

        global_index = None
        if self.use_global:
            global_index = S + len(ctx_tokens)
            g = self.global_token.expand(B, -1, -1) + self.global_cond(z).unsqueeze(1)
            ctx_tokens.append(g)
            ctx_names.append("global")

        if ctx_tokens:
            context = torch.cat(ctx_tokens, dim=1)
            context_mask = torch.ones(B, context.size(1), dtype=torch.bool, device=device)
            all_tokens = torch.cat([component_tokens, context], dim=1)
            all_mask = torch.cat([component_mask, context_mask], dim=1)
        else:
            context = None
            all_tokens, all_mask = component_tokens, component_mask

        meta = {
            "num_context": 0 if context is None else context.size(1),
            "global_index": global_index,
            "context_names": ctx_names,
            "component_len": S,
        }
        return all_tokens, all_mask, meta


class ComponentTokenEmbedder(nn.Module):
    """
    Component token embedder.
    """

    def __init__(
        self, dim, max_type=4, use_type=True, use_amount=True, separate_amount_emb=True
    ):
        super().__init__()
        self.dim = dim
        self.use_type = use_type
        self.use_amount = use_amount
        self.separate_amount_emb = separate_amount_emb

        if use_type:
            self.type_emb = nn.Embedding(max_type, dim)
        if use_amount:
            if separate_amount_emb:
                self.solvent_amount_mlp = ModernMLP(
                    1, dim, dim, num_layers=2, norm_type="identity")
                self.salt_amount_mlp = ModernMLP(
                    1, dim, dim, num_layers=2, norm_type="identity")
            else:
                self.amount_mlp = ModernMLP(
                    1, dim, dim, num_layers=2, norm_type="identity")
        self.norm = nn.LayerNorm(dim)

    def forward(self, h_graph, graph_types, graph_amounts):
        out = h_graph
        if self.use_type:
            out = out + self.type_emb(graph_types.clamp_min(0).clamp_max(3))
        if self.use_amount:
            if self.separate_amount_emb:
                is_solvent = (graph_types == 0).float().unsqueeze(-1)
                is_salt = (graph_types == 1).float().unsqueeze(-1)
                solvent_emb = self.solvent_amount_mlp(graph_amounts.unsqueeze(-1))
                salt_emb = self.salt_amount_mlp(graph_amounts.unsqueeze(-1))
                out = out + solvent_emb * is_solvent + salt_emb * is_salt
            else:
                out = out + self.amount_mlp(graph_amounts.unsqueeze(-1))
        return self.norm(out)


# ================== relation-aware interaction ==================


class RelationAwareInteractionLayer(nn.Module):
    """
    Backward-compatible fallback:
    if relation_type is None, all edges share relation_type=0.
    """

    def __init__(
        self, dim, relation_dim=16, num_relations=8, cond_dim=4,
        dropout=0.0, norm_type="layernorm", use_attention=True
    ):
        super().__init__()
        self.use_attention = use_attention
        self.rel_emb = nn.Embedding(num_relations, relation_dim)
        self.cond_proj = nn.Linear(cond_dim, relation_dim)
        in_dim = 2 * dim + relation_dim + relation_dim
        self.norm = make_norm(dim, norm_type)
        self.msg = nn.Sequential(
            nn.Linear(in_dim, dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim)
        )
        self.gate = nn.Sequential(
            nn.Linear(in_dim, dim // 2),
            nn.SiLU(),
            nn.Linear(dim // 2, 1)
        )
        self.upd = nn.Sequential(
            nn.Linear(2 * dim, dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim)
        )
        self.dropout = nn.Dropout(dropout)
        self.out_norm = make_norm(dim, norm_type)

    def forward(self, h, edge_index, sample_cond=None, edge_sample_index=None,
                relation_type=None):
        if edge_index is None or edge_index.numel() == 0:
            return h, None

        row, col = edge_index
        device = h.device
        E = row.numel()
        if relation_type is None:
            relation_type = torch.zeros(E, dtype=torch.long, device=device)
        rel = self.rel_emb(relation_type.clamp_min(0).clamp_max(self.rel_emb.num_embeddings - 1))

        if sample_cond is None:
            cond = torch.zeros(E, self.cond_proj.in_features, device=device, dtype=h.dtype)
        else:
            if edge_sample_index is None:
                edge_sample_index = torch.zeros(E, dtype=torch.long, device=device)
            cond = sample_cond[edge_sample_index]
        cond = self.cond_proj(cond)

        hn = self.norm(h)
        feat = torch.cat([hn[row], hn[col], rel, cond], dim=-1)
        msg = self.msg(feat)

        alpha = None
        if self.use_attention:
            logits = self.gate(feat).squeeze(-1)
            logits_exp = torch.exp(
                logits - scatter(logits, row, dim=0, dim_size=h.size(0), reduce="max")[row])
            denom = scatter(logits_exp, row, dim=0, dim_size=h.size(0), reduce="sum")[row].clamp_min(1e-8)
            alpha = logits_exp / denom
            msg = msg * alpha.unsqueeze(-1)

        agg = scatter(msg, row, dim=0, dim_size=h.size(0), reduce="sum")
        dh = self.upd(torch.cat([hn, agg], dim=-1))
        out = self.out_norm(h + self.dropout(dh))
        return out, alpha


class AtomInteractiveModule(nn.Module):
    """
    Upgraded GIL-style atom interactive module.
    """

    def __init__(
        self, dim, num_layers=0, relation_dim=16, cond_dim=4,
        dropout=0.0, norm_type="layernorm", use_attention=True
    ):
        super().__init__()
        self.num_layers = int(num_layers)
        self.layers = nn.ModuleList([
            RelationAwareInteractionLayer(
                dim=dim, relation_dim=relation_dim, cond_dim=cond_dim,
                dropout=dropout, norm_type=norm_type, use_attention=use_attention,
            )
            for _ in range(self.num_layers)
        ])

    @property
    def enabled(self):
        return self.num_layers > 0

    def forward(self, atom_repr, inter_edge_index=None, sample_cond=None,
                atom_to_sample=None, relation_type=None):
        if not self.enabled:
            return atom_repr, None

        if inter_edge_index is None:
            raise ValueError(
                "inter_edge_index must be provided when atom_interactive_module is enabled.")

        edge_sample_index = None
        if atom_to_sample is not None and inter_edge_index.numel() > 0:
            row = inter_edge_index[0]
            if row.max().item() < atom_to_sample.numel():
                edge_sample_index = atom_to_sample[row]

        relation_alpha = None
        for layer in self.layers:
            atom_repr, relation_alpha = layer(
                atom_repr, inter_edge_index, sample_cond=sample_cond,
                edge_sample_index=edge_sample_index, relation_type=relation_type,
            )
        return atom_repr, relation_alpha


# ================== heads ==================


class AdditiveEmergentHead(nn.Module):
    """Type-aware additive baseline plus emergent mixture residual."""

    SOLVENT = 0
    SALT = 1

    def __init__(self, dim, hidden_dim=None, dropout=0.0, norm_type="layernorm"):
        super().__init__()
        hidden_dim = hidden_dim or dim
        self.solvent_head = ModernMLP(dim, hidden_dim, 1, num_layers=2,
                                      dropout=dropout, norm_type=norm_type)
        self.salt_head = ModernMLP(dim, hidden_dim, 1, num_layers=2,
                                   dropout=dropout, norm_type=norm_type)
        self.salt_gate = ModernMLP(2, hidden_dim, 1, num_layers=2,
                                   dropout=dropout, norm_type="identity")
        self.emergent_head = ModernMLP(dim, hidden_dim, 1, num_layers=3,
                                       dropout=dropout, norm_type=norm_type)

    def forward(self, component_tokens, component_mask, amount_tensor,
                type_tensor, mixture_state, c):
        type_tensor = type_tensor.long()
        valid = component_mask.bool()
        solvent_mask = valid & (type_tensor == self.SOLVENT)
        salt_mask = valid & (type_tensor == self.SALT)

        solvent_i = self.solvent_head(component_tokens).squeeze(-1)
        solvent_w = amount_tensor.masked_fill(~solvent_mask, 0.0)
        solvent_w = solvent_w / solvent_w.sum(dim=1, keepdim=True).clamp_min(1e-8)
        y_solvent = (solvent_i * solvent_w).sum(dim=1)

        salt_repr = masked_mean(component_tokens, salt_mask)
        salt_present = salt_mask.float().sum(dim=1).clamp(max=1.0)
        salt_base = self.salt_head(salt_repr).squeeze(-1)
        salt_gate = self.salt_gate(torch.stack(
            [c, torch.log1p(c.clamp_min(0))], dim=-1)).squeeze(-1)
        y_salt = salt_present * salt_gate * salt_base

        y_emergent = self.emergent_head(mixture_state).squeeze(-1)
        log_pred = y_solvent + y_salt + y_emergent
        aux = {
            "y_add_solvent": y_solvent.detach(),
            "y_add_salt": y_salt.detach(),
            "y_emg": y_emergent.detach(),
        }
        return log_pred, aux


# ================== SolvMix main model ==================


class SolvMix(nn.Module):
    """SolvMix v6 intuitive."""

    def __init__(
        self,
        num_layer,
        node_input_dim,
        edge_attr_dim,
        hidden_dim,
        num_atten_layer=3,
        num_atten_head=4,
        num_atom_interaction_layer=0,
        seq_len=16,
        dropout=0.05,
        attn_dropout=0.0,
        resid_dropout=0.0,
        ffn_mult=4.0,
        norm_type="layernorm",
        cond_mode="in_context",
        token_pool="global",
        use_type_emb=True,
        use_amount_emb=True,
        use_relation=False,
        use_relation_attn=True,
        relation_dim=16,
        use_decomp=True,
        use_global_token=True,
        atom_mod_mode="gating",
        token_block_mod="none",
        separate_amount_emb=True,
        device="cpu",
    ):
        super().__init__()
        self.d = hidden_dim
        self.seq_len = seq_len
        self.cond_mode = cond_mode
        self.token_pool = token_pool
        self.use_decomp = use_decomp
        self.use_global_token = use_global_token
        self.atom_mod_mode = atom_mod_mode
        self.token_block_mod = token_block_mod
        self.condition_builder = ConditionBuilder()

        self.solv_graph_encoder = ConditionedGraphEncoder(
            num_layer, node_input_dim, edge_attr_dim, hidden_dim,
            cond_dim=5, dropout=dropout, norm_type=norm_type,
            atom_mod_mode=atom_mod_mode,
        )
        self.salt_graph_encoder = ConditionedGraphEncoder(
            num_layer, node_input_dim, edge_attr_dim, hidden_dim,
            cond_dim=5, dropout=dropout, norm_type=norm_type,
            atom_mod_mode=atom_mod_mode,
        )
        self.atom_interactive_module = AtomInteractiveModule(
            dim=hidden_dim,
            num_layers=num_atom_interaction_layer,
            relation_dim=relation_dim,
            cond_dim=4,
            dropout=dropout,
            norm_type=norm_type,
            use_attention=use_relation_attn,
        )
        self.token_embedder = ComponentTokenEmbedder(
            hidden_dim, use_type=use_type_emb, use_amount=use_amount_emb,
            separate_amount_emb=separate_amount_emb
        )
        self.token_film_conditioner = FiLMConditioner(
            hidden_dim, cond_dim=4, hidden_dim=hidden_dim)
        self.context_token_builder = ContextTokenBuilder(
            hidden_dim,
            use_temp=True,
            use_comp=True,
            use_global=use_global_token,
            separate_amount_emb=True,
        )
        self.token_interactive_module = TokenInteractiveModule(
            dim=hidden_dim, num_layers=num_atten_layer, num_heads=num_atten_head,
            ffn_mult=ffn_mult, attn_dropout=attn_dropout, resid_dropout=resid_dropout,
            ffn_dropout=dropout, norm_type=norm_type, return_attn=False,
            block_cond_mode=token_block_mod, cond_dim=4,
        )
        self.mixture_pooler = MaskedAttentionPooling(hidden_dim)
        self.readout = ModernMLP(hidden_dim, hidden_dim, 1, num_layers=3,
                                 dropout=dropout, norm_type=norm_type)
        self.late_concat_projector = ModernMLP(
            hidden_dim + 4, hidden_dim, hidden_dim,
            num_layers=2, dropout=dropout, norm_type=norm_type)
        self.decomp_head = AdditiveEmergentHead(
            hidden_dim, hidden_dim, dropout=dropout, norm_type=norm_type)
        self.to(device)

    def build_global_condition(self, T, c):
        return self.condition_builder.global_condition(T, c)

    def build_component_metadata(
        self, n_graph, g2b_indices, ratios, c, dtype=None,
        graph_types=None, graph_amounts=None
    ):
        return self.condition_builder.build_component_metadata(
            n_graph=n_graph, g2b_indices=g2b_indices, ratios=ratios, c=c,
            dtype=dtype, graph_types=graph_types, graph_amounts=graph_amounts,
        )

    def broadcast_atom_condition(self, T, c, n2b_indices, n2g_indices, graph_amounts):
        return self.condition_builder.atom_condition(
            T, c, n2b_indices, n2g_indices, graph_amounts)

    def graph_to_token_pooling(self, atom_repr, n2g_indices):
        return scatter_mean(atom_repr, n2g_indices, dim=0)

    def _prepare_component_batch(self, h_graph, graph_types, graph_amounts, g2b_indices, batch_size):
        h_graph = self.token_embedder(h_graph, graph_types, graph_amounts)
        component_tokens, component_mask, _ = scatter_batch_with_mask(
            h_graph, g2b_indices, batch_size, self.seq_len
        )
        amount_tensor, _, _ = scatter_scalar_batch_with_mask(
            graph_amounts, g2b_indices, batch_size, self.seq_len
        )
        type_tensor, _, _ = scatter_scalar_batch_with_mask(
            graph_types.float(), g2b_indices, batch_size, self.seq_len
        )
        return component_tokens, component_mask, amount_tensor, type_tensor.long()

    def _condition_tokens_for_mixer(self, component_tokens, component_mask, amount_tensor, T, c, z, type_tensor=None):
        context_meta = {"global_index": None, "component_len": component_tokens.size(1)}
        if self.cond_mode == "film":
            mixer_tokens = self.token_film_conditioner(component_tokens, z)
            mixer_mask = component_mask
        elif self.cond_mode == "in_context":
            mixer_tokens, mixer_mask, context_meta = self.context_token_builder(
                component_tokens, component_mask, T, c, amount_tensor, type_tensor
            )
        elif self.cond_mode in ("none", "late_concat", "adaln_zero"):
            mixer_tokens, mixer_mask = component_tokens, component_mask
        else:
            mixer_tokens, mixer_mask, context_meta = self.context_token_builder(
                component_tokens, component_mask, T, c, amount_tensor, type_tensor
            )
        return mixer_tokens, mixer_mask, context_meta

    def _read_mixture_state(self, token_repr, component_repr, component_mask, context_meta):
        global_index = context_meta.get("global_index", None)
        pool_weights = None
        if self.token_pool == "global" and global_index is not None:
            mixture_state = token_repr[:, global_index, :]
        elif self.token_pool == "mean":
            mixture_state = masked_mean(component_repr, component_mask)
        else:
            mixture_state, pool_weights = self.mixture_pooler(
                component_repr, component_mask)
        return mixture_state, pool_weights

    def forward(self, batch, salt_batch, n2g_indices, n2b_indices, g2b_indices,
                T, c, salt_one_hot=None, ratios=None, graph_types=None,
                graph_amounts=None, inter_edge_index=None, batch_size=None,
                return_state=True, T_override=None, c_override=None,
                ratios_override=None, graph_amounts_override=None):
        if T_override is not None:
            T = T_override
        if c_override is not None:
            c = c_override
        if ratios_override is not None:
            ratios = ratios_override
        if graph_amounts_override is not None:
            graph_amounts = graph_amounts_override

        x_solv, x_salt = batch.x, salt_batch.x
        if batch_size is None:
            batch_size = int(g2b_indices.max().item()) + 1

        n_graph = int(n2g_indices.max().item()) + 1
        graph_types, graph_amounts = self.build_component_metadata(
            n_graph=n_graph,
            g2b_indices=g2b_indices,
            ratios=ratios,
            c=c,
            dtype=x_solv.dtype,
            graph_types=graph_types,
            graph_amounts=graph_amounts,
        )
        atom_cond_all = self.broadcast_atom_condition(
            T, c, n2b_indices, n2g_indices, graph_amounts
        )
        num_solv_atoms = x_solv.size(0)
        atom_cond_solv = atom_cond_all[:num_solv_atoms]
        atom_cond_salt = atom_cond_all[num_solv_atoms:]

        h_atom_solv = self.solv_graph_encoder(
            x_solv, batch.edge_index, batch.edge_attr, cond_node=atom_cond_solv
        )
        h_atom_salt = self.salt_graph_encoder(
            x_salt, salt_batch.edge_index, salt_batch.edge_attr, cond_node=atom_cond_salt
        )
        h_atom = torch.cat([h_atom_solv, h_atom_salt], dim=0)

        z = self.build_global_condition(T, c)
        h_atom, relation_alpha = self.atom_interactive_module(
            atom_repr=h_atom,
            inter_edge_index=inter_edge_index,
            sample_cond=z,
            atom_to_sample=n2b_indices,
            relation_type=None,
        )

        h_graph = self.graph_to_token_pooling(h_atom, n2g_indices)
        component_tokens, component_mask, amount_tensor, type_tensor = self._prepare_component_batch(
            h_graph, graph_types, graph_amounts, g2b_indices, batch_size
        )

        mixer_tokens, mixer_mask, context_meta = self._condition_tokens_for_mixer(
            component_tokens, component_mask, amount_tensor, T, c, z, type_tensor
        )
        mixer_cond = z if (
            self.cond_mode == "adaln_zero" or self.token_block_mod == "adaln_zero"
        ) else None
        token_repr, attn_maps = self.token_interactive_module(
            mixer_tokens, mask=mixer_mask, cond=mixer_cond)

        component_len = context_meta.get("component_len", self.seq_len)
        component_repr = token_repr[:, :component_len, :]
        mixture_state, pool_weights = self._read_mixture_state(
            token_repr, component_repr, component_mask, context_meta
        )

        if self.cond_mode == "late_concat":
            mixture_state = self.late_concat_projector(
                torch.cat([mixture_state, z], dim=-1))

        if self.use_decomp:
            log_pred, head_aux = self.decomp_head(
                component_repr, component_mask, amount_tensor, type_tensor, mixture_state, c
            )
        else:
            log_pred = self.readout(mixture_state).squeeze(-1)
            head_aux = {}

        if relation_alpha is not None:
            sparse_reg = relation_alpha.abs().mean()
        else:
            sparse_reg = torch.zeros((), device=log_pred.device, dtype=log_pred.dtype)

        aux = {
            "mixture_state": mixture_state,
            "component_tokens": component_repr,
            "component_mask": component_mask,
            "amount_tensor": amount_tensor,
            "type_tensor": type_tensor,
            "pool_weights": pool_weights,
            "attn_maps": attn_maps,
            "sparse_reg": sparse_reg,
            **head_aux,
        }
        return log_pred, aux
