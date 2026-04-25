import bz2
import gzip
import json
import lzma
import os
from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from rdkit import Chem
from torch.utils.data import Dataset
from torch_geometric.data import Batch, Data

# ---------------------------------------------------------------------------
# Path resolution (works regardless of cwd)
# ---------------------------------------------------------------------------
_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_MODULE_DIR)


def _resolve_compressed_path(path: str) -> str:
    """If the given path does not exist, try compressed variants (.xz, .gz, .bz2)."""
    if os.path.exists(path):
        return path
    for ext in (".xz", ".gz", ".bz2"):
        compressed = path + ext
        if os.path.exists(compressed):
            return compressed
    return path


def _open_json(path: str):
    """Open a (possibly compressed) JSON file for reading text."""
    if path.endswith(".xz"):
        return lzma.open(path, "rt", encoding="utf-8")
    if path.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    if path.endswith(".bz2"):
        return bz2.open(path, "rt", encoding="utf-8")
    return open(path, "r", encoding="utf-8")


_SMILES_MAPPING_PATH = _resolve_compressed_path(
    os.path.join(_PROJECT_DIR, "raw", "smiles_mapping.json")
)

with _open_json(_SMILES_MAPPING_PATH) as f:
    smiles_mapping = json.load(f)

solvent_smiles_map = smiles_mapping["solvents"]
salt_smiles_map = smiles_mapping["salts"]

salt_list = [
    "LiPF6",
    "LiBF4",
    "LiFSI",
    "LiTDI",
    "LiPDI",
    "LiTFSI",
    "LiClO4",
    "LiAsF6",
    "LiBOB",
    "LiCF3SO3",
    "LiBPFPB",
    "LiBMB",
    "LiN(CF3SO2)2",
    "LiCTFSI",
    "LiDFOB",
    "LiFSA",
    "LiFNFSI",
]
_salt_to_idx = {name: i for i, name in enumerate(salt_list)}

# ---------------------------------------------------------------------------
# Atom / bond features
# ---------------------------------------------------------------------------


def get_atom_features_onehot(atom):
    """Generate 152-dim one-hot atom features."""
    atomic_num = atom.GetAtomicNum()
    atomic_num_onehot = F.one_hot(torch.tensor(atomic_num), num_classes=119).float()
    degree = atom.GetDegree()
    degree_onehot = F.one_hot(torch.tensor(degree), num_classes=7).float()
    num_h = atom.GetTotalNumHs()
    h_onehot = F.one_hot(torch.tensor(num_h), num_classes=5).float()
    valence = atom.GetTotalValence()
    valence_onehot = F.one_hot(torch.tensor(valence), num_classes=7).float()
    charge = atom.GetFormalCharge()
    charge_idx = max(0, min(charge + 1, 4))
    charge_onehot = F.one_hot(torch.tensor(charge_idx), num_classes=5).float()
    aromatic = int(atom.GetIsAromatic())
    aromatic_onehot = F.one_hot(torch.tensor(aromatic), num_classes=2).float()
    hyb = atom.GetHybridization().real
    hyb_map = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4}
    hyb_idx = hyb_map.get(int(hyb), 0)
    hyb_onehot = F.one_hot(torch.tensor(hyb_idx), num_classes=5).float()
    ring = int(atom.IsInRing())
    ring_onehot = F.one_hot(torch.tensor(ring), num_classes=2).float()
    return torch.cat(
        [
            atomic_num_onehot,
            degree_onehot,
            h_onehot,
            valence_onehot,
            charge_onehot,
            aromatic_onehot,
            hyb_onehot,
            ring_onehot,
        ]
    )


def smiles_to_pyg_data(smiles: str) -> Optional[Data]:
    """Convert SMILES to PyG Data with 152-dim node features and 13-dim edge features."""
    if not smiles or pd.isna(smiles):
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    atom_feats = []
    for atom in mol.GetAtoms():
        feat = get_atom_features_onehot(atom)
        atom_feats.append(feat)
    x = torch.stack(atom_feats)

    edge_index = []
    edge_attr = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bond_type = int(bond.GetBondType())
        bond_type_map = {1: 0, 2: 1, 3: 2, 12: 3}
        bond_type_idx = bond_type_map.get(bond_type, 0)
        bond_type_onehot = F.one_hot(torch.tensor(bond_type_idx), num_classes=4).float()
        stereo = bond.GetStereo()
        stereo_map = {
            "STEREONONE": 0,
            "STEREOANY": 1,
            "STEREOCIS": 2,
            "STEREOTRANS": 3,
            "STEREOE": 4,
            "STEREOZ": 5,
        }
        stereo_name = str(stereo).split(".")[-1] if "." in str(stereo) else str(stereo)
        stereo_idx = stereo_map.get(stereo_name, 0)
        stereo_onehot = F.one_hot(torch.tensor(stereo_idx), num_classes=6).float()
        is_conjugated = int(bond.GetIsConjugated())
        is_aromatic = int(bond.GetIsAromatic())
        is_in_ring = int(bond.IsInRing())
        edge_feat = torch.cat(
            [
                bond_type_onehot,
                stereo_onehot,
                torch.tensor([is_conjugated, is_aromatic, is_in_ring], dtype=torch.float),
            ]
        )
        edge_index.extend([[i, j], [j, i]])
        edge_attr.extend([edge_feat, edge_feat])

    if len(edge_index) == 0:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, 13), dtype=torch.float)
    else:
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.stack(edge_attr)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


def precompute_all_molecule_graphs() -> Dict[str, Data]:
    """Precompute molecular graphs for all known solvents and salts from smiles_mapping."""
    all_graphs: Dict[str, Data] = {}
    for name, smiles in solvent_smiles_map.items():
        if smiles:
            graph = smiles_to_pyg_data(smiles)
            if graph is not None:
                all_graphs[name] = graph
    for name, smiles in salt_smiles_map.items():
        if smiles:
            graph = smiles_to_pyg_data(smiles)
            if graph is not None:
                all_graphs[name] = graph
    return all_graphs


# ---------------------------------------------------------------------------
# Unified Dataset
# ---------------------------------------------------------------------------


class UnifiedDataset(Dataset):
    """
    读取 raw/ 下统一格式的 JSON 数据集。

    溶剂组分使用 solvent mol ratio；盐使用 salt concentration / mol_ratio。
    两者物理单位和语义不同，collate_fn 会显式返回 graph_types 与
    graph_amounts，避免在模型中通过顺序或 ratios 长度反推。

    支持通过 dataset_name 参数选择数据源。
    处理后的数据会自动缓存到 processed/ 目录，相同分子图在不同样本间共享，
    通过索引映射避免重复存储，显著降低内存占用并加快二次加载速度。
    """

    DATASET_PATHS = {
        "bamboo_mixer": os.path.join(_PROJECT_DIR, "raw", "Bamboo-Mixer_exp_data.json"),
        "edb1": os.path.join(_PROJECT_DIR, "raw", "EDB-1.json"),
        "geomix_calisol": os.path.join(_PROJECT_DIR, "raw", "GeoMix_CALiSol.json"),
        "geomix_diffmix": os.path.join(_PROJECT_DIR, "raw", "GeoMix_DiffMix.json"),
    }

    def __init__(self, dataset_name, max_samples=1e8, device="cpu", force_reprocess=False):
        super().__init__()
        self.dataset_name = dataset_name
        self.max_samples = int(max_samples)
        self.device = device

        dataset_path = self.DATASET_PATHS.get(dataset_name)
        if dataset_path is None:
            raise ValueError(
                f"Unknown dataset_name '{dataset_name}'. "
                f"Available: {list(self.DATASET_PATHS.keys())}"
            )
        dataset_path = _resolve_compressed_path(dataset_path)
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

        processed_dir = os.path.join(_PROJECT_DIR, "processed")
        os.makedirs(processed_dir, exist_ok=True)
        processed_path = os.path.join(processed_dir, f"{dataset_name}.pt")

        need_process = force_reprocess or not os.path.exists(processed_path)
        if not need_process and os.path.exists(dataset_path):
            raw_mtime = os.path.getmtime(dataset_path)
            proc_mtime = os.path.getmtime(processed_path)
            if raw_mtime > proc_mtime:
                need_process = True

        if need_process:
            self.process(dataset_path, processed_path)
        else:
            print(f"Loading processed dataset from {processed_path}")
            cache = torch.load(processed_path, map_location="cpu")
            self.unique_graphs = cache["unique_graphs"]
            self.processed_data = cache["processed_data"]

        self.processed_data = self.processed_data[: self.max_samples]
        print(f"Dataset processed length: {len(self.processed_data)}")

    def process(self, dataset_path, processed_path):
        """
        处理原始数据并缓存到 processed_path。
        通过 unique_graphs 列表去重，processed_data 中只存储索引，
        实现不同样本间相同分子图的高效共享。
        """
        raw_data = self._load_dataset(dataset_path)
        print(f"Dataset raw length: {len(raw_data)}")

        # 预计算所有已知分子图
        graph_cache = precompute_all_molecule_graphs()
        # smiles -> name 反向映射
        smiles_to_name: Dict[str, str] = {}
        for name, smiles in solvent_smiles_map.items():
            if smiles:
                smiles_to_name[smiles] = name
        for name, smiles in salt_smiles_map.items():
            if smiles:
                smiles_to_name[smiles] = name

        # 去重后的图列表和映射
        unique_graphs: List[Data] = []
        graph_key_to_idx: Dict[str, int] = {}

        def _canonical_key(key: str) -> Optional[str]:
            """返回用于去重的 canonical key（优先使用 name）。"""
            if not key:
                return None
            if key in graph_cache:
                return key
            if key in smiles_to_name:
                return smiles_to_name[key]
            return key

        def _get_or_add_graph(key: str) -> Optional[int]:
            """获取或添加图，返回在 unique_graphs 中的索引。"""
            canon = _canonical_key(key)
            if canon is None:
                return None
            if canon in graph_key_to_idx:
                return graph_key_to_idx[canon]

            # 生成图对象
            if key in graph_cache:
                graph = graph_cache[key]
            elif key in smiles_to_name:
                graph = graph_cache.get(smiles_to_name[key])
            else:
                graph = smiles_to_pyg_data(key)

            if graph is None:
                return None

            graph_key_to_idx[canon] = len(unique_graphs)
            unique_graphs.append(graph)
            return graph_key_to_idx[canon]

        processed_data: List[Dict] = []
        dropped = 0
        for item in raw_data:
            solvent_indices = []
            for key in item["solvent_names"]:
                idx = _get_or_add_graph(key)
                if idx is not None:
                    solvent_indices.append(idx)

            salt_idx = _get_or_add_graph(item["salt"])
            if not solvent_indices or salt_idx is None:
                dropped += 1
                continue

            processed_data.append(
                {
                    "y": item["y"],
                    "T": item["T"],
                    "c": item["c"],
                    "salt": item["salt"],
                    "salt_idx": _salt_to_idx.get(
                        item["salt_name"] if item["salt_name"] else item["salt"], -1
                    ),
                    "solvent_graph_indices": solvent_indices,
                    "salt_graph_idx": salt_idx,
                    "solvent_ratios": item["solvent_ratios"],
                }
            )

        print(
            f"Dataset processed length: {len(processed_data)} (dropped {dropped}), "
            f"unique graphs: {len(unique_graphs)}"
        )

        self.unique_graphs = unique_graphs
        self.processed_data = processed_data

        torch.save(
            {"unique_graphs": unique_graphs, "processed_data": processed_data},
            processed_path,
        )
        print(f"Saved processed dataset to {processed_path}")

    def _load_dataset(self, dataset_path):
        dataset_path = _resolve_compressed_path(dataset_path)
        with _open_json(dataset_path) as f:
            raw_data = json.load(f)

        data = []
        for item in raw_data:
            conductivity = float(item.get("conductivity", 0))
            temperature = float(item.get("temperature", 298.15))
            if conductivity <= 0:
                continue

            solvents = []
            solvent_ratios = []
            solvent_names = []

            for s in item.get("solvents", []):
                name = s.get("name", "") or ""
                smiles = s.get("smiles", "") or ""
                ratio = s.get("solvents_mol_ratio", 0)
                if not name and not smiles:
                    continue
                if pd.isna(ratio) or ratio <= 0:
                    continue
                # lookup key: prefer name if available, otherwise smiles
                lookup_key = name if name else smiles
                solvents.append(lookup_key)
                solvent_ratios.append(float(ratio))
                solvent_names.append(lookup_key)

            if not solvents:
                continue

            salt_info = item.get("salts", {})
            # Defensive: handle list-form salts (some datasets may use a list)
            if isinstance(salt_info, list) and salt_info:
                salt_info = salt_info[0]
            elif not isinstance(salt_info, dict):
                salt_info = {}

            salt_name = salt_info.get("name", "") or ""
            salt_smiles = salt_info.get("smiles", "") or ""
            salt_key = salt_name if salt_name else salt_smiles
            salt_mol_ratio = float(salt_info.get("mol_ratio", 0))

            data.append(
                {
                    "solvents": solvents,
                    "solvent_ratios": solvent_ratios,
                    "solvent_names": solvent_names,
                    "salt": salt_key,
                    "salt_name": salt_name,
                    "salt_mol_ratio": salt_mol_ratio,
                    "T": temperature,
                    "y": conductivity,
                    "c": salt_mol_ratio,
                }
            )
        return data

    def __len__(self):
        return len(self.processed_data)

    def __getitem__(self, idx):
        item = self.processed_data[idx]
        return {
            "y": item["y"],
            "T": item["T"],
            "c": item["c"],
            "salt": item["salt"],
            "salt_idx": item["salt_idx"],
            "solvent_graphs": [self.unique_graphs[i] for i in item["solvent_graph_indices"]],
            "salt_graph": self.unique_graphs[item["salt_graph_idx"]],
            "solvent_ratios": item["solvent_ratios"],
        }
