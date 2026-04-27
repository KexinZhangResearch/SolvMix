import datetime
import os
import shutil
from functools import partial

import torch
from torch.utils.data import DataLoader

from .dataset import RDKit2DConductivityDatasetOptimized, solv_mix_collate_fn_optimized


class CachedBatchLoader:
    """将单个 batch 包装成可迭代对象，行为类似 DataLoader"""

    def __init__(self, batch):
        self.batch = batch

    def __iter__(self):
        yield self.batch

    def __len__(self):
        return 1


def build_dataloaders(cfg):
    data_cfg = cfg.data
    dataset = RDKit2DConductivityDatasetOptimized(
        dataset_name=data_cfg.dataset_name,
        max_samples=data_cfg.max_samples,
    )

    total = len(dataset)
    train_size = int(total * data_cfg.train_ratio)
    val_size = int(total * data_cfg.val_ratio)
    test_size = total - train_size - val_size

    train_set, val_set, test_set = torch.utils.data.random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(cfg.seed),
    )

    collate_fn = partial(solv_mix_collate_fn_optimized, seq_len=data_cfg.seq_len)

    if data_cfg.batch_size is None:
        # Full-batch mode
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        cache_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", "cache",
            f"full_batch_{data_cfg.max_samples}_seq{data_cfg.seq_len}_{timestamp}"
        )
        os.makedirs(cache_dir, exist_ok=True)

        def get_cached_batch(subset, name):
            cache_path = os.path.join(cache_dir, f"{name}.pt")
            if os.path.exists(cache_path):
                batch = torch.load(cache_path, map_location='cpu')
                print(f"[v5] Loaded cached {name} batch from {cache_path}")
            else:
                data_list = [subset[i] for i in range(len(subset))]
                batch = collate_fn(data_list)
                torch.save(batch, cache_path)
                print(f"[v5] Saved {name} batch to {cache_path}")
            return CachedBatchLoader(batch)

        train_loader = get_cached_batch(train_set, "train")
        val_loader = get_cached_batch(val_set, "val")
        test_loader = get_cached_batch(test_set, "test")
        print("[v5] Full-batch mode: static structures will be cached during forward")
    else:
        dataloader_args = {
            'batch_size': data_cfg.batch_size,
            'collate_fn': collate_fn,
            'num_workers': data_cfg.num_workers,
            'pin_memory': (data_cfg.num_workers > 0),
            'persistent_workers': (data_cfg.num_workers > 0),
        }
        train_loader = DataLoader(train_set, shuffle=True, **dataloader_args)
        val_loader = DataLoader(val_set, shuffle=False, **dataloader_args)
        test_loader = DataLoader(test_set, shuffle=False, **dataloader_args)
        cache_dir = None

    return train_loader, val_loader, test_loader, cache_dir
