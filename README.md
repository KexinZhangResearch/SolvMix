# SolvMix

SolvMix is a deep learning framework for predicting properties of solvent mixtures (e.g., electrolyte formulations) using graph neural networks. It supports multi-component systems including solvents, salts, and additives, and is built on top of [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/) and [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/).

Supported datasets:

- **CALiSol**
- **DiffMix**
- **EDB-1**
- **Bamboo-Mixer**

***

## Installation

### 1. Create a Conda/Mamba environment (recommended)

```bash
conda create -n solvmix python=3.11
conda activate solvmix
```

> You may also use `mamba` instead of `conda` for faster dependency resolution.

### 2. Install PyTorch and dependencies

Choose the requirements file that matches your CUDA version:

**CUDA 12.1**

```bash
pip install -r requirements-cu121.txt
```

**CUDA 11.8**

```bash
pip install -r requirements-cu118.txt
```

> The `requirements-cu*.txt` files already include index URLs for the PyPI mirror (Tsinghua) and PyTorch/PyG official wheels. If you encounter slow downloads for CUDA-specific wheels, it is usually because `torch==2.1.0+cu121` and `torch-scatter` etc. must be fetched from the PyTorch / PyG official repositories, which are hosted outside mainland China.

### 3. Verify installation

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

***

## Quick Start

### Training

The easiest way to start training is using the provided shell script:

```bash
./trainer.sh
```

Or explicitly specify a config:

```bash
./trainer.sh --config base
```

#### Experiment Configs

Place experiment-specific configs under `configs/experiments/` and inherit from `base`:

```yaml
# configs/experiments/my_exp.yaml
defaults:
  - base

exp_name: my_experiment
data:
  seq_len: 128
train:
  learning_rate: 1.0e-3
```

Run it by name (the loader automatically falls back to `configs/experiments/`):

```bash
./trainer.sh --config my_exp
```

You can also use the full path:

```bash
./trainer.sh --config experiments/my_exp
```

#### Specify GPU(s)

Set the `SOLVMIX_GPU` environment variable before running the script. This internally sets `CUDA_VISIBLE_DEVICES` so PyTorch only sees the selected GPU(s):

```bash
SOLVMIX_GPU=0 ./trainer.sh           # use GPU 0 only
SOLVMIX_GPU=2,3 ./trainer.sh         # use GPUs 2 and 3
```

You can also run the trainer module directly (run from the repository root):

```bash
python trainer.py --config base
```

### Configuration

SolvMix uses [OmegaConf](https://omegaconf.readthedocs.io/) for hierarchical configuration. The default config composition is defined in `configs/base.yaml`:

```yaml
defaults:
  - data: calisol      # choose from calisol / diffmix / edb1 / bamboo_mixer
  - model: default
  - train: default
```

To switch datasets, either edit `configs/base.yaml` or create a new config file and pass it via `--config <name>`.

Key training hyperparameters (in `configs/train/default.yaml`):

- Epochs: 5000
- Learning rate: 3e-4
- Optimizer: AdamW
- Gradient clip: 0.5
- EMA decay: 0.999

### GPU Stress Test (Optional)

A helper script to keep GPUs under continuous load is included:

```bash
./run_gpu.sh              # occupy all GPUs
./run_gpu.sh --gpus 0 1   # occupy specific GPUs
```

***

## Project Structure

```
SolvMix/
├── src/                 # Source code
│   ├── callbacks/       # PyTorch Lightning callbacks (EMA, etc.)
│   ├── dataset.py       # Dataset definition and SMILES processing
│   ├── dataloader.py    # DataLoader and collate function
│   └── model.py         # SolvMix GNN model
├── configs/             # OmegaConf configurations
│   ├── data/            # Dataset configs (CALiSol, DiffMix, EDB-1, Bamboo-Mixer)
│   ├── experiments/     # Experiment-specific configs inheriting from base
│   ├── model/           # Model architecture configs
│   └── train/           # Training hyperparameter configs
├── raw/                 # Raw dataset files and SMILES mappings
├── trainer.py           # PyTorch Lightning training loop
├── trainer.sh           # Training launcher script
├── run_gpu.sh           # GPU stress test script
├── requirements.txt         # CPU-only dependencies
├── requirements-cu118.txt   # CUDA 11.8 dependencies
├── requirements-cu121.txt   # CUDA 12.1 dependencies
└── environment.yml      # Conda environment specification
```

***

## License

See [LICENSE](./LICENSE).
