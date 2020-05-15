import torch
import torch.nn as nn

from models.cgan import Cgan

# This formulation is consistent with the original GAN training objective
class TheoreticalCgan(Cgan):
    def gen_loss(self, gen_logits):
        # -1 to get correct loss formulation
        return (-1.)*self.bce_logits(gen_logits, torch.zeros_like(gen_logits))

# Total Variation
# Note: Very hard to train, gradients saturate because of the tanh:s
class TVCgan(Cgan):
    def disc_loss(self, data_logits, gen_logits):
        return torch.mean(-0.5*torch.tanh(data_logits)) +\
            torch.mean(0.5*torch.tanh(gen_logits))

    def gen_loss(self, gen_logits):
        return torch.mean(-0.5*torch.tanh(gen_logits))

# Kullback-Leibler
# Gradient clipping would likely be useful
class KLCgan(Cgan):
    def disc_loss(self, data_logits, gen_logits):
        return torch.mean(-1.*data_logits) +\
            torch.mean(torch.exp(gen_logits - 1.))

    def gen_loss(self, gen_logits):
        return torch.mean(-1.*gen_logits)

# Reverse Kullback-Leibler
class RKLCgan(Cgan):
    def disc_loss(self, data_logits, gen_logits):
        return torch.mean(torch.exp(data_logits)) +\
            torch.mean(-gen_logits-1.)

    def gen_loss(self, gen_logits):
        return torch.mean(torch.exp(gen_logits))

# Pearson chi-squared
class PearsonCgan(Cgan):
    def disc_loss(self, data_logits, gen_logits):
        return torch.mean(-1.*data_logits) +\
            torch.mean(gen_logits*(0.25*gen_logits + 1.0))

    def gen_loss(self, gen_logits):
        return torch.mean(-1.*gen_logits)

# Neyman chi-squared
class NeymanCgan(Cgan):
    def disc_loss(self, data_logits, gen_logits):
        return torch.mean(-1.+torch.exp(data_logits)) +\
            torch.mean(2.*(1. - torch.exp(0.5*gen_logits)))

    def gen_loss(self, gen_logits):
        return torch.mean(-1.+torch.exp(gen_logits))

# Jensen Shannon Divergence
class JSCgan(Cgan):
    def disc_loss(self, data_logits, gen_logits):
        return (-1.0)*torch.log(torch.tensor(2., device=self.device)) +\
            self.bce_logits(data_logits, torch.ones_like(data_logits)) +\
            (-0.5)*torch.mean(
                    torch.log(1.0 - (2.0)*torch.pow(torch.sigmoid(gen_logits), 2))
                )

    def gen_loss(self, gen_logits):
        return (-0.5)*torch.log(torch.tensor(2., device=self.device)) +\
            self.bce_logits(gen_logits, torch.ones_like(gen_logits))

# Squared Hellinger Distance
class SHCgan(Cgan):
    def disc_loss(self, data_logits, gen_logits):
        return torch.mean(-1.+torch.exp(data_logits)) +\
            torch.mean(-1. + torch.exp((-1.)*gen_logits))

    def gen_loss(self, gen_logits):
        return torch.mean(-1.+torch.exp(gen_logits))

# Least-Square CGAN
class LSCgan(Cgan):
    def disc_loss(self, data_logits, gen_logits):
        return 0.5*(nn.functional.mse_loss(data_logits, torch.ones_like(data_logits))
                + nn.functional.mse_loss(gen_logits, torch.zeros_like(gen_logits)))

    def gen_loss(self, gen_logits):
        return 0.5*nn.functional.mse_loss(gen_logits, torch.ones_like(gen_logits))

cgans = {
    "standard": Cgan,
    "theoretical": TheoreticalCgan,
    "tv": TVCgan,
    "kl": KLCgan,
    "rkl": RKLCgan,
    "pearson": PearsonCgan,
    "neyman": NeymanCgan,
    "js": JSCgan,
    "sh": SHCgan,
    "ls": LSCgan,
}

def build_cgan(config, div_estimator=False):
    if div_estimator:
        return cgans[div_estimator](config, div_estimator)
    else:
        return cgans[config["cgan_type"]](config)

