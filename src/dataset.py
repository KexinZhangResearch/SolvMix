import gzip
import json
import lzma
import os
import pickle
from typing import Dict, Optional

import pandas as pd
import torch
import torch.nn.functional as F
from rdkit import Chem
from torch.utils.data import Dataset
from torch_geometric.data import Batch, Data

# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------
_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_MODULE_DIR)
RAW_DIR = os.path.join(_PROJECT_DIR, "raw")
PROCESSED_DIR = os.path.join(_PROJECT_DIR, "processed")
os.makedirs(PROCESSED_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Salt vocabulary
# ---------------------------------------------------------------------------
salt_list = [
    'LiPF6', 'LiBF4', 'LiFSI', 'LiTDI', 'LiPDI', 'LiTFSI',
    'LiClO4', 'LiAsF6', 'LiBOB', 'LiCF3SO3', 'LiBPFPB', 'LiBMB', 'LiN(CF3SO2)2'
]

_salt_to_idx = {name: i for i, name in enumerate(salt_list)}


# ---------------------------------------------------------------------------
# RDKit graph featurization
# ---------------------------------------------------------------------------
def get_atom_features_onehot(atom):
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
    return torch.cat([
        atomic_num_onehot, degree_onehot, h_onehot, valence_onehot,
        charge_onehot, aromatic_onehot, hyb_onehot, ring_onehot
    ])


def smiles_to_pyg_data(smiles: str) -> Optional[Data]:
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
        stereo_map = {'STEREONONE': 0, 'STEREOANY': 1, 'STEREOCIS': 2,
                      'STEREOTRANS': 3, 'STEREOE': 4, 'STEREOZ': 5}
        stereo_name = str(stereo).split('.')[-1] if '.' in str(stereo) else str(stereo)
        stereo_idx = stereo_map.get(stereo_name, 0)
        stereo_onehot = F.one_hot(torch.tensor(stereo_idx), num_classes=6).float()
        is_conjugated = int(bond.GetIsConjugated())
        is_aromatic = int(bond.GetIsAromatic())
        is_in_ring = int(bond.IsInRing())
        edge_feat = torch.cat([
            bond_type_onehot, stereo_onehot,
            torch.tensor([is_conjugated, is_aromatic, is_in_ring], dtype=torch.float)
        ])
        edge_index.extend([[i, j], [j, i]])
        edge_attr.extend([edge_feat, edge_feat])

    if len(edge_index) == 0:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, 13), dtype=torch.float)
    else:
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.stack(edge_attr)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


def precompute_all_molecule_graphs(solvent_smiles_map: Dict[str, str], salt_smiles_map: Dict[str, str]) -> Dict[str, Data]:
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
# Compressed JSON helper
# ---------------------------------------------------------------------------
def _resolve_data_path(name: str) -> str:
    base = os.path.join(RAW_DIR, name)
    if os.path.exists(base):
        return base
    # Try compressed variants
    for ext in (".xz", ".gz", ".bz2"):
        compressed = base + ext
        if os.path.exists(compressed):
            return compressed
    # Try .json + compression (e.g. GeoMix_CALiSol.json.xz)
    if not name.endswith(".json"):
        json_base = os.path.join(RAW_DIR, name + ".json")
        if os.path.exists(json_base):
            return json_base
        for ext in (".xz", ".gz", ".bz2"):
            compressed = json_base + ext
            if os.path.exists(compressed):
                return compressed
    return base


def _open_json(path: str):
    if path.endswith(".xz"):
        return lzma.open(path, "rt", encoding="utf-8")
    if path.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, "r", encoding="utf-8")


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class RDKit2DConductivityDatasetOptimized(Dataset):
    def __init__(self, dataset_name: str = "GeoMix_CALiSol", max_samples: int = 100_000, device: str = "cpu"):
        super().__init__()
        self.dataset_name = dataset_name
        self.max_samples = int(max_samples)
        self.device = device
        self.raw_data = self.load_dataset()

        solvent_smiles_map: Dict[str, str] = {}
        salt_smiles_map: Dict[str, str] = {}
        for item in self.raw_data:
            for sol in item.get('solvents', []):
                name = sol.get('name', '')
                smiles = sol.get('smiles', '')
                if name and smiles:
                    solvent_smiles_map[name] = smiles
            salt = item.get('salts', {})
            salt_name = salt.get('name', '')
            salt_smiles = salt.get('smiles', '')
            if salt_name and salt_smiles:
                salt_smiles_map[salt_name] = salt_smiles

        self._graph_cache = precompute_all_molecule_graphs(solvent_smiles_map, salt_smiles_map)

        print("Processing data...")
        dataset_process_path = os.path.join(
            PROCESSED_DIR, f"{dataset_name}_rdkit_v5_max{self.max_samples}.pkl"
        )
        if os.path.exists(dataset_process_path):
            with open(dataset_process_path, 'rb') as f:
                self.data = pickle.load(f)
            print(f"Loaded from cache: {dataset_process_path}")
        else:
            self.data = self.process(self.raw_data)
            with open(dataset_process_path, 'wb') as f:
                pickle.dump(self.data, f)
            print(f"Cached to: {dataset_process_path}")
        print(f"dataset total length: {len(self.data)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        solvent_graphs = []
        for name in item['solvent_names']:
            g = self._graph_cache.get(name)
            if g is not None:
                solvent_graphs.append(g)
        salt_graph = self._graph_cache.get(item['salt'], None)
        return {
            'k': item['k'],
            'T': item['T'],
            'c': item['c'],
            'salt': item['salt'],
            'graphs': solvent_graphs,
            'salt_graphs': [salt_graph] if salt_graph is not None else [],
            'ratios': torch.tensor(item['ratios'], dtype=torch.float32),
        }

    def load_dataset(self):
        path = _resolve_data_path(self.dataset_name)
        with _open_json(path) as f:
            data = json.load(f)
        return data[:self.max_samples]

    def process(self, raw_data):
        processed = []
        for item in raw_data:
            conductivity = item.get('conductivity')
            if conductivity is None or pd.isna(conductivity) or float(conductivity) <= 0:
                continue

            solvents = item.get('solvents', [])
            if not solvents:
                continue

            solvent_names = []
            ratios = []
            for sol in solvents:
                name = sol.get('name', '')
                ratio = sol.get('solvents_mol_ratio', 0)
                if name and float(ratio) > 0:
                    solvent_names.append(name)
                    ratios.append(float(ratio))

            if not solvent_names:
                continue

            salt_info = item.get('salts', {})
            salt_name = salt_info.get('name', '')
            if not salt_name:
                continue

            c_val = float(salt_info.get('mol_ratio', 0))
            T_val = float(item.get('temperature', 298.15))
            y_val = float(conductivity)

            processed.append({
                'k': y_val,
                'T': T_val,
                'c': c_val,
                'salt': salt_name,
                'solvent_names': solvent_names,
                'ratios': ratios,
            })
        return processed


# ---------------------------------------------------------------------------
# Collate
# ---------------------------------------------------------------------------
def solv_mix_collate_fn_optimized(batch, device='cpu', seq_len=8):
    batch_size = len(batch)
    y_values = torch.empty(batch_size, dtype=torch.float32)
    T_values = torch.empty(batch_size, dtype=torch.float32)
    c_values = torch.empty(batch_size, dtype=torch.float32)
    salt_one_hot = torch.zeros(batch_size, len(salt_list), dtype=torch.float32)

    for idx, item in enumerate(batch):
        y_values[idx] = item['k']
        T_values[idx] = item['T']
        c_values[idx] = item['c']
        sidx = _salt_to_idx.get(item['salt'], -1)
        if sidx >= 0:
            salt_one_hot[idx, sidx] = 1.0

    all_solvent_graphs = []
    all_salt_graphs = []
    g2b_solvent = []
    g2b_salt = []
    batch_ratios_list = []

    for sample_idx, item in enumerate(batch):
        for g in item['graphs']:
            all_solvent_graphs.append(g)
            g2b_solvent.append(sample_idx)
        batch_ratios_list.extend(item['ratios'].tolist())
        for g in item['salt_graphs']:
            all_salt_graphs.append(g)
            g2b_salt.append(sample_idx)

    batch_ratios = torch.tensor(batch_ratios_list, dtype=torch.float32)

    if all_solvent_graphs:
        batch_solvents = Batch.from_data_list(all_solvent_graphs)
        ptr = batch_solvents.ptr
        n2g_solvent = torch.arange(len(ptr) - 1, dtype=torch.long).repeat_interleave((ptr[1:] - ptr[:-1]))
        g2b_solvent_tensor = torch.tensor(g2b_solvent, dtype=torch.long)
        n2b_solvent = g2b_solvent_tensor[n2g_solvent]
    else:
        batch_solvents = Data(
            x=torch.zeros(1, 152, dtype=torch.float32),
            edge_index=torch.zeros(2, 0, dtype=torch.long),
            edge_attr=torch.zeros(0, 13, dtype=torch.float32)
        )
        n2g_solvent = torch.tensor([0], dtype=torch.long)
        g2b_solvent_tensor = torch.tensor([0], dtype=torch.long)
        n2b_solvent = torch.tensor([0], dtype=torch.long)

    if all_salt_graphs:
        batch_salts = Batch.from_data_list(all_salt_graphs)
        salt_ptr = batch_salts.ptr
        n2g_salt = torch.arange(len(salt_ptr) - 1, dtype=torch.long).repeat_interleave((salt_ptr[1:] - salt_ptr[:-1]))
        g2b_salt_tensor = torch.tensor(g2b_salt, dtype=torch.long)
        n2b_salt = g2b_salt_tensor[n2g_salt]
    else:
        batch_salts = Data(
            x=torch.zeros(1, 152, dtype=torch.float32),
            edge_index=torch.zeros(2, 0, dtype=torch.long),
            edge_attr=torch.zeros(0, 13, dtype=torch.float32)
        )
        n2g_salt = torch.tensor([], dtype=torch.long)
        g2b_salt_tensor = torch.tensor([], dtype=torch.long)
        n2b_salt = torch.tensor([], dtype=torch.long)

    if n2g_solvent.numel() > 0:
        n2g_max = int(n2g_solvent.max().item())
    else:
        n2g_max = -1

    if n2g_salt.numel() > 0:
        n2g_salt_shifted = n2g_salt + n2g_max + 1
        n2g_indices = torch.cat([n2g_solvent, n2g_salt_shifted], dim=0)
    else:
        n2g_indices = n2g_solvent.clone()

    n2b_indices = torch.cat([n2b_solvent, n2b_salt], dim=0)
    g2b_indices = torch.cat([g2b_solvent_tensor, g2b_salt_tensor], dim=0)

    batch_solvents.n2g_indices = n2g_solvent
    batch_solvents.n2b_indices = n2b_solvent
    batch_solvents.g2b_indices = g2b_solvent_tensor
    batch_salts.n2g_indices = n2g_salt
    batch_salts.n2b_indices = n2b_salt
    batch_salts.g2b_indices = g2b_salt_tensor

    # Precompute pos_idx for scatter_batch
    counts = g2b_indices.bincount(minlength=batch_size)
    max_mols = int(counts.max().item())
    seq_len = max(seq_len, max_mols)

    sorted_idx = torch.argsort(g2b_indices)
    sorted_samples = g2b_indices[sorted_idx]
    sorted_pos = torch.arange(len(g2b_indices))
    group_starts = torch.cat([
        torch.tensor([True]),
        sorted_samples[1:] != sorted_samples[:-1]
    ])
    group_start_pos = torch.zeros_like(sorted_pos)
    group_start_pos[group_starts] = sorted_pos[group_starts]
    group_start_expanded = group_start_pos.cummax(0).values
    sorted_pos_within_group = sorted_pos - group_start_expanded
    pos_idx = torch.empty_like(sorted_pos_within_group)
    pos_idx[sorted_idx] = sorted_pos_within_group
    pos_idx = pos_idx.clamp(max=seq_len - 1)

    num_solvent_graphs = len(all_solvent_graphs)

    return {
        'y': y_values.to(device, non_blocking=True),
        'T': T_values.to(device, non_blocking=True),
        'c': c_values.to(device, non_blocking=True),
        'salt_one_hot': salt_one_hot.to(device, non_blocking=True),
        'batch': batch_solvents.to(device),
        'salt_batch': batch_salts.to(device),
        'n2g_indices': n2g_indices.to(device),
        'n2b_indices': n2b_indices.to(device),
        'g2b_indices': g2b_indices.to(device),
        'ratios': batch_ratios.to(device, non_blocking=True),
        'pos_idx': pos_idx.to(device),
        'seq_len': seq_len,
        'num_solvent_graphs': num_solvent_graphs,
    }
