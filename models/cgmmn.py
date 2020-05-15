import torch
import math
import wandb

from models.nn import NN
import models.noise_dists as nds
import utils

# CGMMN from Ren et al.
class CGMMN(NN):
    def __init__(self, config):
        super().__init__(config)

        self.feed_noise = True

        self.mmd_lambda = config["mmd_lambda"]
        self.sqrt_loss = config["sqrt_loss"]

        # If no specific scales parameter given for x or y, fall back to shared scale
        if not config["mmd_scales_x"]:
            config["mmd_scales_x"] = config["mmd_scales"]
        if not config["mmd_scales_y"]:
            config["mmd_scales_y"] = config["mmd_scales"]

        self.mmd_scales_x = torch.tensor(config["mmd_scales_x"], device=self.device)
        self.mmd_scales_y = torch.tensor(config["mmd_scales_y"], device=self.device)

        # Noise distribution to sample from
        self.noise_dist = nds.get_noise_dist(config)

        # Do not shuffle data to get same batches each epoch
        self.shuffle = False
        self.inverses = [] # List to save inverse matrices in

    def compute_dims(self, config, spec):
        # Simplest case, takes just x in
        spec["in_dim"] = config["x_dim"] + config["noise_dim"]
        spec["out_dim"] = config["y_dim"]
        spec["x_dim"] = config["x_dim"]
        spec["other_dim"] = config["noise_dim"]

    def process_net_input(self, x_batch, fixed_noise=False, **kwarg):
        # Append noise vector
        if fixed_noise:
            noise = self.noise_dist.sample([1]).to(self.device)
            noise_batch = noise.repeat((x_batch.shape[0],1))
        else:
            noise_batch = self.noise_dist.sample([x_batch.shape[0]]).to(self.device)

        return torch.cat((x_batch, noise_batch), dim=1)

    def gram_matrix(self, batch1, batch2, scales):
        n_b2 = batch2.shape[0]
        batch_1_matrix = batch1.unsqueeze(1).repeat(1,n_b2,1)

        diff_matrix = batch_1_matrix - batch2 # Shape (n_b1, n_b2, dim_y)
        distance_matrix = torch.sum(torch.pow(diff_matrix, 2), dim=2) # L2-norm of diff

        n_scales = scales.shape[0]
        repeated_distances = distance_matrix.unsqueeze(0)/scales.view(
                n_scales, 1,1) # Shape: (n_scales, n_b1, n_b2)

        normalizers = torch.rsqrt(2.*math.pi*scales)
        gram_matrix = torch.sum(
                normalizers.view(n_scales,1,1)*\
                torch.exp((-0.5)*repeated_distances)
            , dim=0)

        return gram_matrix

    # Conditional MMD loss
    def loss(self, logits, y_batch, x_batch, batch_i, **kwargs):
        # logits is actual sample
        # Loss follows eq. 2 in original paper
        K = self.gram_matrix(x_batch, x_batch, self.mmd_scales_x)
        L_s = self.gram_matrix(logits, logits, self.mmd_scales_y)
        L_d = self.gram_matrix(y_batch, y_batch, self.mmd_scales_y)
        L_ds = self.gram_matrix(y_batch, logits, self.mmd_scales_y)

        if batch_i < len(self.inverses):
            K_inv = self.inverses[batch_i]
        elif batch_i == len(self.inverses):
            # First epoch, inverse not yet computed
            eye = torch.eye(x_batch.shape[0], device=self.device)
            K_tilde = K + self.mmd_lambda*eye
            wandb.log({"x_gram_det": torch.det(K_tilde)})
            K_inv = torch.inverse(K_tilde)

            # Store computed inverse for batch
            self.inverses.append(K_inv)
        else:
            assert False, "Batch order is off, no precomputed inverse found"

        K_K_inv = K@K_inv

        c1 = torch.trace(K_K_inv@L_d@K_inv)
        c2 = torch.trace(K_K_inv@L_s@K_inv)
        c3 = (-2.)*torch.trace(K_K_inv@L_ds@K_inv)

        loss = c1 + c2 + c3

        if self.sqrt_loss:
            return torch.sqrt(loss)
        else:
            return loss

    @torch.no_grad()
    def get_pdf(self, x, n_samples=100, **kwargs):
        # Make sure a good kernel_scale has been infered
        assert self.kernel_scale, "No kernel scale stored for CGAN"

        # Use KDE to estimate pdf
        if not type(x) == torch.Tensor:
            x = torch.tensor([x])

        xs = x.repeat(n_samples, 1)
        samples = self.sample(xs).to("cpu")
        return utils.kde_pdf(samples, self.kernel_scale)

    @torch.no_grad()
    def sample(self, xs, batch_size=None, fixed_noise=False, **kwargs):
        n = xs.shape[0]
        xs = xs.to(self.device)
        net_input = self.process_net_input(xs, fixed_noise=fixed_noise)

        if batch_size and (n > batch_size):
            # Batch sampling process
            batches = torch.split(net_input, batch_size, dim=0)

            sample_list = []
            for input_batch in batches:
                sample_batch = self.net(input_batch)
                sample_list.append(sample_batch)

            samples = torch.cat(sample_list, dim=0)

        else:
            samples = self.net(net_input)

        return samples

    @torch.no_grad()
    def eval(self, dataset, config, use_best_kernel_scale=False):
        # Eval using KDE
        ks = None # None means try many kernel scales

        # Compute Mean Absolute Error, minimized by sample median
        if use_best_kernel_scale:
            # Make sure a good kernel_scale has been infered
            assert self.kernel_scale, "No kernel scale stored for CGAN"
            ks = self.kernel_scale

        evaluation_vals, best_scale = utils.kde_eval(self, dataset, config,
                kernel_scale=ks)
        self.kernel_scale = best_scale
        return evaluation_vals

