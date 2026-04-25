import json
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
_SMILES_MAPPING_PATH = os.path.join(_MODULE_DIR, "raw", "smiles_mapping.json")

with open(_SMILES_MAPPING_PATH, "r") as f:
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
    """

    DATASET_PATHS = {
        "bamboo_mixer": os.path.join(_MODULE_DIR, "raw", "Bamboo-Mixer_exp_data.json"),
        "edb1": os.path.join(_MODULE_DIR, "raw", "EDB-1.json"),
        "geomix_calisol": os.path.join(_MODULE_DIR, "raw", "GeoMix_CALiSol.json"),
        "geomix_diffmix": os.path.join(_MODULE_DIR, "raw", "GeoMix_DiffMix.json"),
    }

    def __init__(self, dataset_name, max_samples=1e8, device="cpu"):
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
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

        self.data = self._load_dataset(dataset_path)
        self.data = self.data[: self.max_samples]
        print(f"Dataset raw length: {len(self.data)}")

        # 预计算所有已知分子图
        self._graph_cache = precompute_all_molecule_graphs()
        # smiles -> name 反向映射，用于 name 为 null 时查找
        self._smiles_to_name: Dict[str, str] = {}
        for name, smiles in solvent_smiles_map.items():
            if smiles:
                self._smiles_to_name[smiles] = name
        for name, smiles in salt_smiles_map.items():
            if smiles:
                self._smiles_to_name[smiles] = name
        # 动态缓存：遇到不在预计算 cache 中的 smiles 时实时生成
        self._dynamic_cache: Dict[str, Optional[Data]] = {}

        # 在初始化时就把每个样本的图对象准备好
        self.processed_data: List[Dict] = []
        dropped = 0
        for item in self.data:
            solvent_graphs = []
            for key in item["solvent_names"]:
                g = self._get_graph(key)
                if g is not None:
                    solvent_graphs.append(g)
            salt_graph = self._get_graph(item["salt"])
            if not solvent_graphs or salt_graph is None:
                dropped += 1
                continue

            self.processed_data.append(
                {
                    "y": item["y"],
                    "T": item["T"],
                    "c": item["c"],
                    "salt": item["salt"],
                    "salt_idx": _salt_to_idx.get(
                        item["salt_name"] if item["salt_name"] else item["salt"], -1
                    ),
                    "solvent_graphs": solvent_graphs,
                    "salt_graph": salt_graph,
                    "solvent_ratios": item["solvent_ratios"],
                }
            )
        print(
            f"Dataset processed length: {len(self.processed_data)} (dropped {dropped})"
        )

    def _get_graph(self, key: str) -> Optional[Data]:
        """根据 name 或 smiles 获取分子图。"""
        if not key:
            return None
        # 1. 直接按 name 查找
        if key in self._graph_cache:
            return self._graph_cache[key]
        # 2. 按 smiles 查找反向映射
        if key in self._smiles_to_name:
            name = self._smiles_to_name[key]
            if name in self._graph_cache:
                return self._graph_cache[name]
        # 3. 尝试将 key 作为新 smiles 实时解析
        if key not in self._dynamic_cache:
            self._dynamic_cache[key] = smiles_to_pyg_data(key)
        return self._dynamic_cache[key]

    def _load_dataset(self, dataset_path):
        with open(dataset_path, "r") as f:
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
        return self.processed_data[idx]
