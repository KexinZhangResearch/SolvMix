from functools import partial
from typing import Tuple

import torch
from torch.utils.data import DataLoader
from torch_geometric.data import Batch, Data

from .dataset import UnifiedDataset


def unified_collate_fn(batch, device="cpu"):
    """
    Collate for SolvMix.

    Explicitly returns:
      graph_types:
        0 = solvent
        1 = salt
        2 = additive / other
        3 = unknown

      graph_amounts:
        solvent graph -> solvent mol ratio
        salt graph    -> salt concentration / mol_ratio
    """
    batch_size = len(batch)

    y_values = torch.empty(batch_size, dtype=torch.float32)
    T_values = torch.empty(batch_size, dtype=torch.float32)
    c_values = torch.empty(batch_size, dtype=torch.float32)
    from .dataset import salt_list

    salt_one_hot = torch.zeros(batch_size, len(salt_list), dtype=torch.float32)

    for idx, item in enumerate(batch):
        y_values[idx] = float(item["y"])
        T_values[idx] = float(item["T"])
        c_values[idx] = float(item["c"])
        sidx = int(item.get("salt_idx", -1))
        if sidx >= 0:
            salt_one_hot[idx, sidx] = 1.0

    all_solvent_graphs = []
    all_salt_graphs = []
    g2b_solvent = []
    g2b_salt = []
    solvent_amounts = []
    salt_amounts = []

    for sample_idx, item in enumerate(batch):
        solvent_graphs = item["solvent_graphs"]
        solvent_ratios = item["solvent_ratios"]

        if len(solvent_graphs) != len(solvent_ratios):
            raise ValueError(
                f"Mismatch in sample {sample_idx}: "
                f"{len(solvent_graphs)} solvent graphs but "
                f"{len(solvent_ratios)} solvent ratios."
            )

        for g, ratio in zip(solvent_graphs, solvent_ratios):
            all_solvent_graphs.append(g)
            g2b_solvent.append(sample_idx)
            solvent_amounts.append(float(ratio))

        if item["salt_graph"] is not None:
            all_salt_graphs.append(item["salt_graph"])
            g2b_salt.append(sample_idx)
            # Salt amount is the sample-level salt concentration / mol_ratio.
            salt_amounts.append(float(item["c"]))

    batch_ratios = torch.tensor(solvent_amounts, dtype=torch.float32)

    if all_solvent_graphs:
        batch_solvents = Batch.from_data_list(all_solvent_graphs)
        ptr = batch_solvents.ptr
        n2g_solvent = torch.arange(len(ptr) - 1).repeat_interleave(
            ptr[1:] - ptr[:-1]
        )
        g2b_solvent_tensor = torch.tensor(g2b_solvent, dtype=torch.long)
        n2b_solvent = g2b_solvent_tensor[n2g_solvent]
    else:
        batch_solvents = Data(
            x=torch.zeros(1, 152, dtype=torch.float32),
            edge_index=torch.zeros(2, 0, dtype=torch.long),
            edge_attr=torch.zeros(0, 13, dtype=torch.float32),
        )
        ptr = torch.tensor([0, 1], dtype=torch.long)
        n2g_solvent = torch.tensor([0], dtype=torch.long)
        g2b_solvent_tensor = torch.tensor([0], dtype=torch.long)
        n2b_solvent = torch.tensor([0], dtype=torch.long)
        solvent_amounts = [0.0]

    if all_salt_graphs:
        batch_salts = Batch.from_data_list(all_salt_graphs)
        salt_ptr = batch_salts.ptr
        n2g_salt_local = torch.arange(len(salt_ptr) - 1).repeat_interleave(
            salt_ptr[1:] - salt_ptr[:-1]
        )
        g2b_salt_tensor = torch.tensor(g2b_salt, dtype=torch.long)
        n2b_salt = g2b_salt_tensor[n2g_salt_local]
    else:
        batch_salts = Data(
            x=torch.zeros(1, 152, dtype=torch.float32),
            edge_index=torch.zeros(2, 0, dtype=torch.long),
            edge_attr=torch.zeros(0, 13, dtype=torch.float32),
        )
        salt_ptr = torch.tensor([0, 1], dtype=torch.long)
        n2g_salt_local = torch.tensor([0], dtype=torch.long)
        g2b_salt_tensor = torch.tensor([0], dtype=torch.long)
        n2b_salt = torch.tensor([0], dtype=torch.long)
        salt_amounts = [0.0]

    num_solvent_graphs = len(g2b_solvent_tensor)
    num_salt_graphs = len(g2b_salt_tensor)
    num_graphs = num_solvent_graphs + num_salt_graphs

    n2g_indices = torch.cat(
        [n2g_solvent, n2g_salt_local + num_solvent_graphs], dim=0
    )
    n2b_indices = torch.cat([n2b_solvent, n2b_salt], dim=0)
    g2b_indices = torch.cat([g2b_solvent_tensor, g2b_salt_tensor], dim=0)

    graph_types = torch.cat(
        [
            torch.zeros(num_solvent_graphs, dtype=torch.long),
            torch.ones(num_salt_graphs, dtype=torch.long),
        ],
        dim=0,
    )
    graph_amounts = torch.cat(
        [
            torch.tensor(solvent_amounts, dtype=torch.float32),
            torch.tensor(salt_amounts, dtype=torch.float32),
        ],
        dim=0,
    )

    if graph_types.numel() != num_graphs or graph_amounts.numel() != num_graphs:
        raise RuntimeError(
            f"Bad graph metadata: num_graphs={num_graphs}, "
            f"graph_types={graph_types.numel()}, graph_amounts={graph_amounts.numel()}."
        )

    # ---------- CPU 上预计算 inter_edge_index（无 GPU 同步） ----------
    from collections import defaultdict

    ranges = []  # (atom_start, atom_end, batch_idx)
    for g_idx, b_idx in enumerate(g2b_solvent):
        ranges.append((int(ptr[g_idx]), int(ptr[g_idx + 1]), b_idx))

    solvent_num_atoms = int(batch_solvents.x.size(0))
    offset = solvent_num_atoms
    for g_idx, b_idx in enumerate(g2b_salt):
        ranges.append(
            (offset + int(salt_ptr[g_idx]), offset + int(salt_ptr[g_idx + 1]), b_idx)
        )

    batch_to_ranges = defaultdict(list)
    for start, end, b_idx in ranges:
        batch_to_ranges[b_idx].append((start, end))

    row_list, col_list = [], []
    for _b_idx, range_list in batch_to_ranges.items():
        if len(range_list) <= 1:
            continue
        for i in range(len(range_list)):
            s_i, e_i = range_list[i]
            if s_i >= e_i:
                continue
            atoms_i = torch.arange(s_i, e_i, dtype=torch.long)
            n_i = atoms_i.numel()
            if n_i == 0:
                continue
            for j in range(i + 1, len(range_list)):
                s_j, e_j = range_list[j]
                if s_j >= e_j:
                    continue
                atoms_j = torch.arange(s_j, e_j, dtype=torch.long)
                n_j = atoms_j.numel()
                if n_j == 0:
                    continue
                row_list.append(atoms_i.repeat_interleave(n_j))
                col_list.append(atoms_j.repeat(n_i))
                row_list.append(atoms_j.repeat_interleave(n_i))
                col_list.append(atoms_i.repeat(n_j))

    if row_list:
        inter_edge_index = torch.stack(
            [torch.cat(row_list), torch.cat(col_list)], dim=0
        )
    else:
        inter_edge_index = torch.zeros((2, 0), dtype=torch.long)

    return {
        "y": y_values,
        "T": T_values,
        "c": c_values,
        "salt_one_hot": salt_one_hot,
        "batch": batch_solvents,
        "salt_batch": batch_salts,
        "n2g_indices": n2g_indices,
        "n2b_indices": n2b_indices,
        "g2b_indices": g2b_indices,
        "ratios": batch_ratios,  # backward compatibility only
        "graph_types": graph_types,
        "graph_amounts": graph_amounts,
        "inter_edge_index": inter_edge_index,
        "batch_size": batch_size,
    }


def build_dataloaders(cfg) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build train / val / test DataLoaders from a unified config.
    Expected cfg fields under cfg.data:
        dataset_name, max_samples, batch_size, num_workers, seq_len,
        train_ratio, val_ratio
    """
    data_cfg = cfg.data

    dataset = UnifiedDataset(
        dataset_name=data_cfg.dataset_name,
        max_samples=data_cfg.max_samples,
    )

    train_ratio = float(data_cfg.get("train_ratio", 0.7))
    val_ratio = float(data_cfg.get("val_ratio", 0.2))
    total = len(dataset)
    train_size = int(total * train_ratio)
    val_size = int(total * val_ratio)
    test_size = total - train_size - val_size

    train_set, val_set, test_set = torch.utils.data.random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(int(cfg.get("seed", 42))),
    )

    num_workers = int(data_cfg.get("num_workers", 0))
    loader_kwargs = dict(
        collate_fn=unified_collate_fn,
        num_workers=num_workers,
        pin_memory=(num_workers > 0),
        persistent_workers=(num_workers > 0),
    )
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = 4

    make_loader = partial(
        DataLoader, batch_size=int(data_cfg.batch_size), **loader_kwargs
    )
    train_loader = make_loader(train_set, shuffle=True)
    val_loader = make_loader(val_set, shuffle=False)
    test_loader = make_loader(test_set, shuffle=False)
    return train_loader, val_loader, test_loader
