import torch
import torch.nn as nn
import wandb

from models.cgmmn import CGMMN
import models.networks as networks

# Joint GMMN model
# Relies on the NN base model through CGMMN
class GMMN(CGMMN):
    def __init__(self, config):
        super().__init__(config)

        # Kernel parameter optimization
        self.x_scaling = nn.Parameter(torch.sqrt(self.mmd_scales_x[0])*torch.ones(
            (1, config["x_dim"]),device=self.device))
        self.y_scaling = nn.Parameter(torch.sqrt(self.mmd_scales_y[0])*torch.ones(
            (1, config["y_dim"]),device=self.device))

        self.kernel_opt = networks.optimizers[config["optimizer"]](
                [self.x_scaling, self.y_scaling], lr=config["kernel_lr"])

        # Reset bandwidth
        self.mmd_scales_x = torch.ones(1, device=self.device)
        self.mmd_scales_y = torch.ones(1, device=self.device)

    def compute_mmd_loss(self, logits, y_batch, x_batch):
        # logits are actual samples

        # Apply component wise scaling (bandwidth)
        logits = logits*self.y_scaling
        y_batch = y_batch*self.y_scaling
        x_batch = x_batch*self.x_scaling

        K_x = self.gram_matrix(x_batch, x_batch, self.mmd_scales_x)
        L_s = self.gram_matrix(logits, logits, self.mmd_scales_y)
        L_ds = self.gram_matrix(y_batch, logits, self.mmd_scales_y)
        L_d = self.gram_matrix(y_batch, y_batch, self.mmd_scales_y)

        loss = torch.trace(K_x@L_s) + torch.trace(K_x@L_d) + (-2.)*torch.trace(K_x@L_ds)
        return loss

    # Joint MMD loss
    def loss(self, logits, y_batch, x_batch, **kwargs):
        # Compute loss first to optimize bandwidth
        self.kernel_opt.zero_grad()
        # Detach tensors since it is unnecessary to backprop through generator
        first_loss = (-1.)*self.compute_mmd_loss(
                logits.detach(), y_batch.detach(), x_batch.detach())
        first_loss.backward()
        self.kernel_opt.step()

        # Second loss is for generator training
        second_loss = self.compute_mmd_loss(logits, y_batch, x_batch)
        return second_loss

    @torch.no_grad()
    def eval(self, dataset, config, use_best_kernel_scale=False):
        # Add in logging of kernel scales when evaluating model
        wandb.log({
            "x_scale": torch.mean(self.x_scaling),
            "y_scale": torch.mean(self.y_scaling),
            })

        # Same eval (KDE) as CGMMN
        return super().eval(dataset, config, use_best_kernel_scale=use_best_kernel_scale)

