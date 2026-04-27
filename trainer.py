import glob
import os

import hydra
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from src.callbacks import EfficientEMACallback
from src.dataloader import build_dataloaders
from src.metrics import regression_metrics
from src.model import SolvMix


# ================== Lightning module ==================


class SolvMixWrapper(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters(OmegaConf.to_container(cfg, resolve=True))

        m_cfg = cfg.model
        self.model = SolvMix(
            num_gnn_blocks=m_cfg.num_gnn_blocks,
            node_input_dim=152,
            edge_attr_dim=13,
            hidden_dim=m_cfg.hidden_dim,
            num_mlp_layer=m_cfg.num_mlp_layer,
            num_token_blocks=m_cfg.num_token_blocks,
            num_atten_head=m_cfg.num_atten_head,
            num_atom_blocks=m_cfg.num_atom_blocks,
            seq_len=cfg.data.seq_len,
            device="cpu",
            if_c_gate=m_cfg.if_c_gate,
        )
        self.loss_fn = torch.nn.MSELoss()
        self.mae_fn = torch.nn.L1Loss()
        self.validation_step_outputs = []
        self.test_step_outputs = []

    def forward(self, batch):
        return self.model(
            batch["batch"],
            batch["salt_batch"],
            batch["n2g_indices"],
            batch["n2b_indices"],
            batch["g2b_indices"],
            batch["T"],
            batch["c"],
            batch.get("ratios"),
            batch.get("pos_idx"),
            batch.get("seq_len"),
        )

    def training_step(self, batch, batch_idx):
        log_pred = self(batch)
        log_target = torch.log1p(batch["y"])
        loss = self.loss_fn(log_pred, log_target)
        self.log("train/loss", loss, prog_bar=True, batch_size=batch["y"].size(0))
        if self.trainer.optimizers:
            lr = self.trainer.optimizers[0].param_groups[0]["lr"]
            self.log("lr", lr, prog_bar=False, batch_size=batch["y"].size(0))
        return loss

    def validation_step(self, batch, batch_idx):
        log_pred = self(batch)
        pred = torch.exp(log_pred) - 1
        loss = self.loss_fn(pred, batch["y"])
        self.log("val/loss", loss, sync_dist=True, batch_size=batch["y"].size(0))
        self.validation_step_outputs.append({"pred": pred.detach(), "target": batch["y"].detach()})
        return {"pred": pred.detach(), "target": batch["y"].detach()}

    def on_validation_epoch_end(self):
        if self.validation_step_outputs:
            preds = torch.cat([x["pred"] for x in self.validation_step_outputs])
            targets = torch.cat([x["target"] for x in self.validation_step_outputs])
            for name, value in regression_metrics(preds, targets).items():
                self.log(f"val/{name}", value, sync_dist=True, prog_bar=(name in {"r2", "pearson"}))
        self.validation_step_outputs.clear()

    def test_step(self, batch, batch_idx):
        log_pred = self(batch)
        pred = torch.exp(log_pred) - 1
        loss = self.loss_fn(pred, batch["y"])
        self.log("test/loss", loss, sync_dist=True, batch_size=batch["y"].size(0))
        self.test_step_outputs.append({"pred": pred.detach(), "target": batch["y"].detach()})
        return {"pred": pred.detach(), "target": batch["y"].detach()}

    def on_test_epoch_end(self):
        if self.test_step_outputs:
            preds = torch.cat([x["pred"] for x in self.test_step_outputs])
            targets = torch.cat([x["target"] for x in self.test_step_outputs])
            for name, value in regression_metrics(preds, targets).items():
                self.log(f"test/{name}", value, sync_dist=True)
            # Log histograms
            if hasattr(self, "logger") and self.logger is not None:
                try:
                    import wandb as _wandb
                    _wandb.log({
                        "test/pred_hist": _wandb.Histogram(preds.detach().cpu().numpy()),
                        "test/target_hist": _wandb.Histogram(targets.detach().cpu().numpy()),
                        "test/error_hist": _wandb.Histogram((preds - targets).detach().cpu().numpy()),
                    })
                except Exception:
                    pass
        self.test_step_outputs.clear()

    def configure_optimizers(self):
        t_cfg = self.cfg.train
        decay_params = []
        no_decay_params = []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if "norm" in name.lower() or name.endswith("bias"):
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        optimizer_group = [
            {"params": decay_params, "weight_decay": t_cfg.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]
        opt_class = torch.optim.AdamW if t_cfg.optimizer == "AdamW" else torch.optim.Adam
        optimizer = opt_class(optimizer_group, lr=t_cfg.learning_rate)

        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1e-5,
            end_factor=1.0,
            total_iters=t_cfg.warmup_steps,
        )
        exponential_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=t_cfg.lr_gamma,
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, exponential_scheduler],
            milestones=[t_cfg.warmup_steps],
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]


# ================== Runner ==================
# Resolve config dir relative to this script so it works regardless of cwd
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_CONFIG_PATH = os.path.join(_SCRIPT_DIR, "src", "configs")


@hydra.main(config_path=_CONFIG_PATH, config_name="base", version_base=None)
def main(cfg: DictConfig) -> None:
    if getattr(cfg, "wandb_api_key", None):
        os.environ["WANDB_API_KEY"] = cfg.wandb_api_key

    if cfg.data.num_workers > 0:
        import torch.multiprocessing as mp
        try:
            mp.set_start_method("spawn", force=True)
        except RuntimeError:
            pass

    pl.seed_everything(cfg.seed)

    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.benchmark = True

    train_loader, val_loader, test_loader, cache_dir = build_dataloaders(cfg)

    _solv_mix_root = os.path.dirname(os.path.abspath(__file__))
    ckpt_dir = os.path.join(_solv_mix_root, "wandb", "checkpoints", cfg.exp_name)
    os.makedirs(ckpt_dir, exist_ok=True)

    checkpoint_callback = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename="best",
        monitor="val/loss",
        mode="min",
        save_top_k=1,
        auto_insert_metric_name=False,
    )

    callbacks = [checkpoint_callback]
    t_cfg = cfg.train
    if t_cfg.use_ema:
        callbacks.append(EfficientEMACallback(decay=t_cfg.ema_decay))

    _solv_mix_root = os.path.dirname(os.path.abspath(__file__))
    wandb_root = os.path.join(_solv_mix_root, "wandb")
    os.makedirs(wandb_root, exist_ok=True)

    import datetime as _dt
    run_id = f"{cfg.exp_name}_{_dt.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    logger = WandbLogger(
        project=cfg.proj_name,
        name=cfg.exp_name,
        save_dir=wandb_root,
        version=run_id,
        offline=getattr(cfg, "wandb_offline", False),
    )

    trainer_kwargs = dict(
        max_epochs=t_cfg.epochs,
        logger=logger,
        callbacks=callbacks,
        precision=t_cfg.precision,
        log_every_n_steps=10,
        check_val_every_n_epoch=t_cfg.test_interval,
        gradient_clip_algorithm="norm",
    )
    if t_cfg.devices > 0:
        trainer_kwargs["accelerator"] = "gpu"
        trainer_kwargs["devices"] = t_cfg.devices
    else:
        trainer_kwargs["accelerator"] = "cpu"
        trainer_kwargs["devices"] = 1
    if t_cfg.grad_clip_val and t_cfg.grad_clip_val > 0:
        trainer_kwargs["gradient_clip_val"] = t_cfg.grad_clip_val

    trainer = pl.Trainer(**trainer_kwargs)

    model = SolvMixWrapper(cfg)

    if cfg.data.batch_size is None:
        model.model.enable_cache()

    if cfg.compile and hasattr(torch, "compile"):
        try:
            model.model = torch.compile(model.model, mode="reduce-overhead")
            print("[torch.compile] Model compiled with mode='reduce-overhead'")
        except Exception as e:
            print(f"[torch.compile] Warning: compilation failed, falling back to eager mode. Error: {e}")

    wandb_subdir = os.path.join(wandb_root, "wandb")
    pre_existing = set(os.listdir(wandb_subdir)) if os.path.exists(wandb_subdir) else set()

    if cfg.test_only:
        ckpt_path = os.path.join(ckpt_dir, "best.ckpt")
        if not os.path.exists(ckpt_path):
            ckpts = glob.glob(os.path.join(ckpt_dir, "best*.ckpt"))
            if ckpts:
                ckpt_path = sorted(ckpts)[-1]
            else:
                raise FileNotFoundError(f"No checkpoint found in {ckpt_dir}")
        trainer.test(model, dataloaders=test_loader, ckpt_path=ckpt_path)
    else:
        trainer.fit(model, train_loader, val_loader)
        trainer.test(dataloaders=test_loader, ckpt_path="best")

    if cache_dir is not None:
        import shutil
        print(f"[v5] Cleaning up full-batch cache: {cache_dir}")
        shutil.rmtree(cache_dir, ignore_errors=True)

    post_existing = set(os.listdir(wandb_subdir)) if os.path.exists(wandb_subdir) else set()
    new_dirs = [
        d for d in (post_existing - pre_existing)
        if os.path.isdir(os.path.join(wandb_subdir, d))
    ]
    if len(new_dirs) == 1:
        old_name = new_dirs[0]
        if old_name.startswith("offline-run-"):
            mode_str = "offline-run"
        elif old_name.startswith("run-"):
            mode_str = "run"
        else:
            mode_str = "run"
        timestamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        new_name = f"{cfg.exp_name}--{timestamp}--{mode_str}"
        old_path = os.path.join(wandb_subdir, old_name)
        new_path = os.path.join(wandb_subdir, new_name)
        if old_path != new_path and os.path.exists(old_path):
            os.rename(old_path, new_path)
            latest_link = os.path.join(wandb_subdir, "latest-run")
            if os.path.islink(latest_link) and os.readlink(latest_link) == old_name:
                os.remove(latest_link)
                os.symlink(new_name, latest_link)
            print(f"[wandb] Renamed run log dir: {old_name} -> {new_name}")


if __name__ == "__main__":
    main()
