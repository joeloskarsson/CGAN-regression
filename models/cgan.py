import os
import torch
import numpy as np
import torch.utils.data as torch_data
import torch.nn as nn
import torch.optim.lr_scheduler as schedulers
import wandb
import math
import shutil

from tabular_dataset import TabularDataset
import models.networks as networks
import utils
import constants
import models.noise_dists as nds
import models.spec_reader as cgan_spec

# CGAN base model
class Cgan:
    def __init__(self, config, div_estimator=False):
        if div_estimator:
            assert config["eval_cgan"], "No eval CGAN specification specified"
            spec = cgan_spec.read_cgan_spec(config["eval_cgan"])
        else:
            assert config["cgan_nets"], "No CGAN network specification specified"
            spec = cgan_spec.read_cgan_spec(config["cgan_nets"])
            wandb.config.cgan_spec = spec

        # Add x,y,noise dimensions to network specs
        spec["gen_network"]["x_dim"] = config["x_dim"]
        spec["gen_network"]["other_dim"] = config["noise_dim"]
        spec["gen_network"]["out_dim"] = config["y_dim"]
        spec["disc_network"]["x_dim"] = config["x_dim"]
        spec["disc_network"]["other_dim"] = config["y_dim"]
        spec["disc_network"]["out_dim"] = 1

        self.device = config["device"]
        self.kernel_scale = None # run ll evaluation to set this correctly

        self.disc = networks.build_network(spec["disc_network"]).to(config["device"])
        if div_estimator:
            self.divergence = div_estimator
        else:
            self.gen = networks.build_network(spec["gen_network"]).to(config["device"])

            self.noise_dist = nds.get_noise_dist(config)

            # Restore model parameters
            if config["restore"]:
                restored_path = utils.parse_restore(config["restore"])
                self.load_params_from_file(restored_path)
            elif config["restore_file"]:
                self.load_params_from_file(config["restore_file"])

        # Binary cross entropy loss that can be reused
        self.bce_logits = nn.BCEWithLogitsLoss(reduction="mean")

        # Set training parameters for G and D from general training options
        # (if not specific ones specified)
        for opt in ["lr", "optimizer", "lr_decay"]:
            for net in ["gen", "disc"]:
                net_option = "{}_{}".format(net, opt)
                if (config[net_option] == None):
                    config[net_option] = config[opt] # Set to general option

    def load_params_from_file(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.disc.load_state_dict(checkpoint["disc"])

        if "gen" in checkpoint:
            self.gen.load_state_dict(checkpoint["gen"])

    def train(self, train_set, config, val_func=None):
        tab_dataset = TabularDataset(train_set)
        n_samples = len(tab_dataset)
        train_loader = torch_data.DataLoader(tab_dataset,
                batch_size=config["batch_size"], shuffle=True,
                num_workers=constants.LOAD_WORKERS)

        # Set models to train mode
        self.gen.train()
        self.disc.train()

        # Keep track of best so far
        best_ll = None # best validation log-likelihood
        best_mae = None # MAE where LL is best (not necessarily best MAE)
        best_epoch_i = None
        best_save_path = os.path.join(wandb.run.dir,
                            constants.BEST_PARAMS_FILE ) # Path to save best params to

        # Optimizers (see GAN-hacks)
        gen_opt = networks.optimizers[config["gen_optimizer"]](
                self.gen.parameters(), lr=config["gen_lr"])
        disc_opt = networks.optimizers[config["disc_optimizer"]](
                self.disc.parameters(), lr=config["disc_lr"])

        # returns multiplicative factor, not new learning rate
        gen_mult_func = (lambda x: config["gen_lr_decay"])
        disc_mult_func = (lambda x: config["disc_lr_decay"])
        gen_scheduler = schedulers.MultiplicativeLR(gen_opt, lr_lambda=gen_mult_func)
        disc_scheduler = schedulers.MultiplicativeLR(disc_opt, lr_lambda=disc_mult_func)

        for epoch_i in range(config["epochs"]):
            epoch_disc_loss = []
            epoch_gen_loss = []
            epoch_fooling = []

            for batch_i, (x_batch, data_batch) in enumerate(train_loader):
                batch_size = data_batch.shape[0]

                # Send to correct device
                x_batch = x_batch.to(config["device"])
                data_batch = data_batch.to(config["device"])

                disc_opt.zero_grad()

                # Sample noise
                noise_batch = self.noise_dist.sample([batch_size]).to(config["device"])

                # Sample from generator
                gen_input = torch.cat((x_batch, noise_batch), dim=1)
                gen_batch = self.gen(gen_input)

                # Train discriminator
                data_logits = self.disc(torch.cat((x_batch, data_batch), dim=1))
                gen_logits = self.disc(torch.cat((x_batch, gen_batch), dim=1))
                disc_loss = self.disc_loss(data_logits, gen_logits)

                disc_loss.backward()
                if config["clip_grad"]:
                    nn.utils.clip_grad_norm_(self.disc.parameters(), config["clip_grad"])
                disc_opt.step()

                gen_opt.zero_grad()

                # Train generator ("new_" here just means part of G training steps)
                n_gen_samples = batch_size*config["gen_samples"]
                new_noise_batch = self.noise_dist.sample([n_gen_samples]).to(
                        config["device"])

                if config["gen_samples"] > 1:
                    # Repeat each x sample an amount of times
                    # to get multiple generator samples for it
                    x_batch_repeated = torch.repeat_interleave(x_batch,
                            config["gen_samples"], dim=0)
                else:
                    x_batch_repeated = x_batch

                new_gen_input = torch.cat((x_batch_repeated, new_noise_batch), dim=1)
                new_gen_batch = self.gen(new_gen_input)
                new_gen_logits = self.disc(
                        torch.cat((x_batch_repeated, new_gen_batch), dim=1))

                gen_loss = self.gen_loss(new_gen_logits)

                gen_loss.backward()
                if config["clip_grad"]:
                    nn.utils.clip_grad_norm_(self.gen.parameters(), config["clip_grad"])
                gen_opt.step()

                # Store loss
                batch_fooling = torch.mean(torch.sigmoid(new_gen_logits))
                epoch_fooling.append(batch_fooling.item())
                epoch_disc_loss.append(disc_loss.item())
                epoch_gen_loss.append(gen_loss.item())

            # Log stats for epoch
            wandb.log({
                "epoch": epoch_i,
                "discriminator_loss": np.mean(epoch_disc_loss),
                "generator_loss": np.mean(epoch_gen_loss),
                "fooling": np.mean(epoch_fooling),
            })
            if val_func and ((epoch_i+1) % config["val_interval"] == 0):
                # Validate
                evaluation_vals = val_func(self, epoch_i)

                if (best_epoch_i == None) or (evaluation_vals["ll"]> best_ll):
                    best_ll = evaluation_vals["ll"]
                    best_mae = evaluation_vals["mae"]
                    best_epoch_i = epoch_i

                    # Save model parameters for best epoch only
                    model_params = {
                        "gen": self.gen.state_dict(),
                        "disc": self.disc.state_dict(),
                    }
                    torch.save(model_params, best_save_path)

            gen_scheduler.step()
            disc_scheduler.step()

        wandb.run.summary["best_epoch"] = best_epoch_i # Save best epoch index to wandb
        wandb.run.summary["log_likelihood"] = best_ll
        wandb.run.summary["mae"] = best_mae

        # Restore best parameters to model (for future testing etc.)
        self.load_params_from_file(best_save_path)

    def disc_loss(self, data_logits, gen_logits):
        # logit arguments are direct outputs from discriminator
        return self.bce_logits(data_logits, torch.ones_like(data_logits)) +\
                self.bce_logits(gen_logits, torch.zeros_like(gen_logits))

    def gen_loss(self, gen_logits):
        # logit arguments are direct outputs from discriminator
        # Practical loss, -log(D) trick
        return self.bce_logits(gen_logits, torch.ones_like(gen_logits))

    @torch.no_grad()
    def get_pdf(self, x, n_samples=100):
        # Make sure a good kernel_scale has been infered
        assert self.kernel_scale, "No kernel scale stored for CGAN"

        # Use KDE (Parzen window) to estimate pdf from samples
        # h_squared is length scale parameter
        if not type(x) == torch.Tensor:
            x = torch.tensor([x])

        xs = x.repeat(n_samples, 1)
        samples = self.sample(xs).to("cpu")
        return utils.kde_pdf(samples, self.kernel_scale)

    @torch.no_grad()
    def sample(self, xs, batch_size=None, fixed_noise=False):
        n = xs.shape[0]
        xs = xs.to(self.device)

        if fixed_noise:
            noise = self.noise_dist.sample([1])
            noise_sample = noise.repeat(n, 1)
        else:
            noise_sample = self.noise_dist.sample([n])
        noise_sample = noise_sample.to(self.device)

        if batch_size and (n > batch_size):
            # Batch sampling process
            batch_iterator = zip(
                torch.split(xs, batch_size, dim=0),
                torch.split(noise_sample, batch_size, dim=0)
            )

            sample_list = []
            for x_batch, noise_batch in batch_iterator:
                gen_input = torch.cat((x_batch, noise_batch), dim=1)
                sample_batch = self.gen(gen_input)
                sample_list.append(sample_batch)

            samples = torch.cat(sample_list, dim=0)

        else:
            gen_input = torch.cat((xs, noise_sample), dim=1)
            samples = self.gen(gen_input)

        return samples

    @torch.no_grad()
    def eval(self, dataset, config, use_best_kernel_scale=False):
        ks = None # None means try many kernel scales

        # Compute Mean Absolute Error, minimized by sample median
        if use_best_kernel_scale:
            # Make sure a good kernel_scale has been inferred
            assert self.kernel_scale, "No kernel scale stored for CGAN"
            ks = self.kernel_scale

        evaluation_vals, best_scale = utils.kde_eval(self, dataset, config,
                kernel_scale=ks)
        self.kernel_scale = best_scale
        return evaluation_vals

    # Train only discriminator using two sets of samples
    # To estimate divergence between (source) distributions of samples
    def train_discriminator(self, config, real_train_set, gen_train_set,
            real_val_set, gen_val_set):

        self.disc.train()

        # Training objects
        optimizer = networks.optimizers[config["disc_optimizer"]](
                self.disc.parameters(), lr=config["disc_lr"])
        mult_func = (lambda x: config["disc_lr_decay"])
        scheduler = schedulers.MultiplicativeLR(optimizer, lr_lambda=mult_func)

        real_tab_dataset = TabularDataset(real_train_set)
        gen_tab_dataset = TabularDataset(gen_train_set)
        real_loader = torch_data.DataLoader(real_tab_dataset,
                batch_size=config["batch_size"], shuffle=True,
                num_workers=constants.LOAD_WORKERS)
        gen_loader = torch_data.DataLoader(gen_tab_dataset,
                batch_size=config["batch_size"], shuffle=True,
                num_workers=constants.LOAD_WORKERS)

        # Highest divergence (on val.-set) so far
        best_divergence = None
        div_save_path = os.path.join(wandb.run.dir, "best_{}_params.pt".format(
            self.divergence)) # Path to save best params to

        for epoch_i in range(config["epochs"]):
            epoch_loss = []

            # Train one epoch
            for (x_real, real_batch), (x_gen, gen_batch) in zip(real_loader, gen_loader):
                optimizer.zero_grad()

                # Concat. and move all to correct device
                real_input = torch.cat((x_real, real_batch), dim=1).to(self.device)
                gen_input = torch.cat((x_gen, gen_batch), dim=1).to(self.device)

                real_logits = self.disc(real_input)
                gen_logits = self.disc(gen_input)

                loss = self.disc_loss(real_logits, gen_logits)
                loss.backward()

                if config["clip_grad"]:
                    nn.utils.clip_grad_norm_(self.disc.parameters(), config["clip_grad"])

                optimizer.step()
                epoch_loss.append(loss.item())

            wandb.log({
                "{}_epoch_i".format(self.divergence): epoch_i,
                "{}_train_divergence".format(self.divergence): (-1.)*np.mean(epoch_loss),
            })

            # Evaluate
            if (epoch_i+1) % config["val_interval"] == 0:
                val_divergence = self.compute_divergence(real_val_set, gen_val_set,
                        config["eval_batch_size"])
                wandb.log({
                    "{}_val_divergence".format(self.divergence): val_divergence,
                })

                if (best_divergence == None) or (val_divergence > best_divergence):
                    # Best divergence so far, save parameters
                    model_params = {
                        "disc": self.disc.state_dict(),
                    }
                    torch.save(model_params, div_save_path)

                    best_divergence = val_divergence

            scheduler.step()

        # Restore best parameters (early stopping)
        self.load_params_from_file(div_save_path)

    # Estimate divergence using discriminator and two sets of samples
    @torch.no_grad()
    def compute_divergence(self, real_dataset, gen_dataset, batch_size):
        real_tab_dataset = TabularDataset(real_dataset)
        gen_tab_dataset = TabularDataset(gen_dataset)
        real_loader = torch_data.DataLoader(real_tab_dataset,
                batch_size=batch_size, shuffle=False,
                num_workers=constants.LOAD_WORKERS)
        gen_loader = torch_data.DataLoader(gen_tab_dataset,
                batch_size=batch_size, shuffle=False,
                num_workers=constants.LOAD_WORKERS)

        total_loss = 0.0
        for (x_real, real_batch), (x_gen, gen_batch) in zip(real_loader, gen_loader):
            # Concat. and move all to correct device
            real_input = torch.cat((x_real, real_batch), dim=1).to(self.device)
            gen_input = torch.cat((x_gen, gen_batch), dim=1).to(self.device)

            real_logits = self.disc(real_input)
            gen_logits = self.disc(gen_input)

            mean_loss = self.disc_loss(real_logits, gen_logits)
            total_loss += mean_loss.item()*x_real.shape[0]

        return (-1.)*(total_loss/len(real_tab_dataset))

