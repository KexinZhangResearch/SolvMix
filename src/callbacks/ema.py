# Copyright 2025 AlQuraishi Laboratory
# Licensed under the Apache License, Version 2.0
# Extracted from models/utils/pl_ema.py

import torch
import torch.nn as nn
import pytorch_lightning as pl


class EfficientEMACallback(pl.Callback):
    """
    高效的 EMA Callback：
    1. 追踪 named_parameters() + floating point buffers（含 BN running stats），
       保证 validation/test 时参数与统计量完全匹配，避免损失波动。
    2. update / swap / restore 均使用原地操作（mul_ / add_ / copy_），避免大量临时张量。
    """

    def __init__(self, decay=0.999):
        super().__init__()
        self.decay = decay
        self.ema_params = {}
        self.ema_buffers = {}
        self._backup_params = {}
        self._backup_buffers = {}

    def on_fit_start(self, trainer, pl_module):
        self.ema_params = {
            name: param.detach().clone()
            for name, param in pl_module.model.named_parameters()
            if param.requires_grad
        }
        # 同步追踪 floating point buffers（如 BN running_mean / running_var）
        self.ema_buffers = {
            name: buffer.detach().clone()
            for name, buffer in pl_module.model.named_buffers()
            if torch.is_floating_point(buffer)
        }

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        with torch.no_grad():
            decay = self.decay
            one_minus_decay = 1.0 - decay

            # --- params: 收集成 list 后用 _foreach 批量更新 ---
            ema_params_list = []
            cur_params_list = []
            for name, param in pl_module.model.named_parameters():
                if name in self.ema_params and param.requires_grad:
                    ema_params_list.append(self.ema_params[name])
                    cur_params_list.append(param.data)

            if ema_params_list:
                if hasattr(torch, '_foreach_mul') and hasattr(torch, '_foreach_add'):
                    torch._foreach_mul_(ema_params_list, decay)
                    torch._foreach_add_(ema_params_list, cur_params_list, alpha=one_minus_decay)
                else:
                    for ema_p, cur_p in zip(ema_params_list, cur_params_list):
                        ema_p.mul_(decay).add_(cur_p, alpha=one_minus_decay)

            # --- buffers: 同样处理 ---
            ema_buffers_list = []
            cur_buffers_list = []
            for name, buffer in pl_module.model.named_buffers():
                if name in self.ema_buffers and torch.is_floating_point(buffer):
                    ema_buffers_list.append(self.ema_buffers[name])
                    cur_buffers_list.append(buffer.data)

            if ema_buffers_list:
                if hasattr(torch, '_foreach_mul') and hasattr(torch, '_foreach_add'):
                    torch._foreach_mul_(ema_buffers_list, decay)
                    torch._foreach_add_(ema_buffers_list, cur_buffers_list, alpha=one_minus_decay)
                else:
                    for ema_b, cur_b in zip(ema_buffers_list, cur_buffers_list):
                        ema_b.mul_(decay).add_(cur_b, alpha=one_minus_decay)

    def _swap_to_ema(self, pl_module):
        """将模型参数与 floating point buffers 替换为 EMA 版本，并备份当前状态。"""
        self._backup_params = {
            name: param.detach().clone()
            for name, param in pl_module.model.named_parameters()
            if name in self.ema_params
        }
        self._backup_buffers = {
            name: buffer.detach().clone()
            for name, buffer in pl_module.model.named_buffers()
            if name in self.ema_buffers
        }
        with torch.no_grad():
            for name, param in pl_module.model.named_parameters():
                if name in self.ema_params:
                    param.copy_(self.ema_params[name])
            for name, buffer in pl_module.model.named_buffers():
                if name in self.ema_buffers:
                    buffer.copy_(self.ema_buffers[name])

    def _restore_backup(self, pl_module):
        """从备份恢复模型参数与 buffers。"""
        with torch.no_grad():
            for name, param in pl_module.model.named_parameters():
                if name in self._backup_params:
                    param.copy_(self._backup_params[name])
            for name, buffer in pl_module.model.named_buffers():
                if name in self._backup_buffers:
                    buffer.copy_(self._backup_buffers[name])
        self._backup_params = {}
        self._backup_buffers = {}

    def on_validation_epoch_start(self, trainer, pl_module):
        self._swap_to_ema(pl_module)
        pl_module.model.eval()

    def on_validation_epoch_end(self, trainer, pl_module):
        self._restore_backup(pl_module)
        pl_module.model.train()

    def on_test_epoch_start(self, trainer, pl_module):
        self._swap_to_ema(pl_module)
        pl_module.model.eval()

    def on_test_epoch_end(self, trainer, pl_module):
        self._restore_backup(pl_module)
        pl_module.model.eval()

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        checkpoint["ema"] = {"params": self.ema_params, "buffers": self.ema_buffers}

    def on_load_checkpoint(self, trainer, pl_module, checkpoint):
        if "ema" in checkpoint:
            self.ema_params = checkpoint["ema"]["params"]
            self.ema_buffers = checkpoint["ema"]["buffers"]
            # 确保设备一致
            device = next(pl_module.model.parameters()).device
            for k in self.ema_params:
                if isinstance(self.ema_params[k], torch.Tensor):
                    self.ema_params[k] = self.ema_params[k].to(device)
            for k in self.ema_buffers:
                if isinstance(self.ema_buffers[k], torch.Tensor):
                    self.ema_buffers[k] = self.ema_buffers[k].to(device)
