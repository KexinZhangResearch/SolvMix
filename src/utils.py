import torch


def scatter_batch(feature, sample_indices, pos_idx, batch_size, seq_len):
    device = feature.device
    hidden_dim = feature.size(-1)
    out = torch.zeros(batch_size, seq_len, hidden_dim, device=device)
    out_flat = out.view(batch_size * seq_len, hidden_dim)
    indices = sample_indices * seq_len + pos_idx
    out_flat.index_add_(0, indices, feature)
    return out


def get_intergraph_edge_index(n2g, n2b):
    """
    稀疏实现：避免构造 N×N 稠密矩阵。
    为减少 tiny-kernel launch 开销，全程在 CPU 完成，最后一次性搬上 GPU。
    """
    device = n2g.device
    n2g_cpu = n2g.cpu()
    n2b_cpu = n2b.cpu()
    num_nodes = n2g_cpu.size(0)
    if num_nodes == 0:
        return torch.zeros((2, 0), dtype=torch.long, device=device)

    sort_idx = torch.argsort(n2b_cpu)
    n2g_sorted = n2g_cpu[sort_idx]
    n2b_sorted = n2b_cpu[sort_idx]

    boundaries = torch.cat([
        torch.tensor([0]),
        torch.where(n2b_sorted[1:] != n2b_sorted[:-1])[0] + 1,
        torch.tensor([num_nodes])
    ]).tolist()

    rows, cols = [], []

    for i in range(len(boundaries) - 1):
        start, end = boundaries[i], boundaries[i + 1]
        if end - start <= 1:
            continue

        batch_nodes = sort_idx[start:end]
        batch_g = n2g_sorted[start:end]

        g_boundaries = torch.cat([
            torch.tensor([0]),
            torch.where(batch_g[1:] != batch_g[:-1])[0] + 1,
            torch.tensor([end - start])
        ]).tolist()

        num_graphs = len(g_boundaries) - 1
        if num_graphs <= 1:
            continue

        graph_nodes = []
        for j in range(num_graphs):
            gs, ge = g_boundaries[j], g_boundaries[j + 1]
            graph_nodes.append(batch_nodes[gs:ge])

        for p in range(num_graphs):
            nodes_p = graph_nodes[p]
            n_p = nodes_p.size(0)
            if n_p == 0:
                continue
            for q in range(p + 1, num_graphs):
                nodes_q = graph_nodes[q]
                n_q = nodes_q.size(0)
                if n_q == 0:
                    continue

                row_pq = nodes_p.repeat_interleave(n_q)
                col_pq = nodes_q.repeat(n_p)
                rows.append(row_pq)
                cols.append(col_pq)
                rows.append(col_pq)
                cols.append(row_pq)

    if not rows:
        return torch.zeros((2, 0), dtype=torch.long, device=device)

    row = torch.cat(rows)
    col = torch.cat(cols)
    edge_index = torch.stack([row, col], dim=0).to(device, non_blocking=True)
    return edge_index
