import torch
import numpy as np
import torch.utils.data as torch_data
import torch.optim.lr_scheduler as schedulers
import wandb
import os
import shutil

from tabular_dataset import TabularDataset
import models.networks as nets
import models.spec_reader as reader
import constants
import models.networks as networks
import utils

# Generic Neural Network model class with training loop
class NN():
    def __init__(self, config):
        assert config["network"], "No network specification specified"
        spec = reader.read_nn_spec(config["network"])
        self.device = config["device"]

        # Compute network dimensions
        self.compute_dims(config, spec)

        # Instantiate network
        self.net = nets.build_network(spec).to(config["device"])
        self.register_buffers(config)

        # Restore model parameters
        if config["restore"]:
            restored_path = utils.parse_restore(config["restore"])
            self.load_params_from_file(restored_path)
        elif config["restore_file"]:
            self.load_params_from_file(config["restore_file"])

        # Shuffle data per default (can be overridden by sub-models)
        self.shuffle = True

    def compute_dims(self, config, spec):
        # Simplest case, takes just x in
        spec["in_dim"] = config["x_dim"]
        spec["out_dim"] = config["y_dim"]

    # Buffers need to be registered before loading network parameters
    def register_buffers(self, config):
        pass

    def train(self, train_set, config, val_func=None):
        tab_dataset = TabularDataset(train_set)
        n_samples = len(tab_dataset)
        train_loader = torch_data.DataLoader(tab_dataset,
                batch_size=config["batch_size"], shuffle=self.shuffle,
                num_workers=constants.LOAD_WORKERS)

        # Set network to train mode
        self.net.train()

        # Keep track of best so far
        best_ll = None # best validation score
        best_mae = None
        best_epoch_i = None
        best_save_path = os.path.join(wandb.run.dir,
                        constants.BEST_PARAMS_FILE ) # Path to save best params to

        # Optimizer
        opt = networks.optimizers[config["optimizer"]](
                self.net.parameters(), lr=config["lr"], weight_decay=config["l2_reg"])

        mult_func = (lambda x: config["lr_decay"])
        scheduler = schedulers.MultiplicativeLR(opt, lr_lambda=mult_func)

        for epoch_i in range(config["epochs"]):
            epoch_loss = []

            for batch_i, (x_batch, y_batch) in enumerate(train_loader):
                batch_size = x_batch.shape[0]

                # Send to correct device
                x_batch = x_batch.to(config["device"])
                y_batch = y_batch.to(config["device"])

                opt.zero_grad()

                # Train network
                net_inputs = self.process_net_input(x_batch, y_batch=y_batch)
                logits = self.net(net_inputs)
                loss = self.loss(logits, y_batch, x_batch=x_batch, batch_i=batch_i)

                loss.backward()
                opt.step()

                # Store loss
                epoch_loss.append(loss.item())

            # Log epoch stats
            wandb.log({
                "epoch": epoch_i,
                "loss": np.mean(epoch_loss)
            })

            if val_func and ((epoch_i+1) % config["val_interval"] == 0):
                evaluation_vals = val_func(self, epoch_i)

                if (best_epoch_i == None) or (evaluation_vals["ll"] > best_ll):
                    best_ll = evaluation_vals["ll"]
                    best_mae = evaluation_vals["mae"]
                    best_epoch_i = epoch_i

                    # Save model parameters for best epoch only
                    model_params = self.net.state_dict()
                    torch.save(model_params, best_save_path)

            scheduler.step()

        # Perform possible additional training
        self.post_training(train_set, config)

        wandb.run.summary["best_epoch"] = best_epoch_i # Save best epoch index to wandb
        wandb.run.summary["log_likelihood"] = best_ll
        wandb.run.summary["mae"] = best_mae

        # Restore best parameters to model (for future testing etc.)
        self.load_params_from_file(best_save_path)

    def process_net_input(self, x_batch, **kwarg):
        return x_batch

    def load_params_from_file(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.net.load_state_dict(checkpoint)

    # Prepare x for pdf functions
    def prepare_x_pdf(self, x):
        if type(x) == torch.Tensor:
            x_batch = x.to(self.device)
        elif type(x) == float:
            x_batch = torch.tensor([x], device=self.device)
        else:
            x_batch = torch.tensor(x, device=self.device)

        x_batch = x_batch.unsqueeze(0)

        return x_batch

    def loss(self, logits, y_batch):
        raise NotImplementedError("No loss implemented for generic neural network model")

    # For additional tasks performed after training
    # Left for sub-models to implement if needed
    def post_training(self, train_set, config):
        pass

    @torch.no_grad()
    def sample_batched(self, xs, batch_size):
        xs = xs.to(self.device)

        if batch_size:
            # Batch sampling process
            batches = torch.split(xs, batch_size, dim=0)

            logits_list = []
            for x_batch in batches:
                logits_batch = self.net(x_batch)
                logits_list.append(logits_batch)

            logits = torch.cat(logits_list, dim=0)
        else:
            logits = self.net(xs)

        return logits

