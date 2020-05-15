import torch
import torch.distributions as tdists

def get_gaussian(config):
    return tdists.multivariate_normal.MultivariateNormal(
            torch.zeros(config["noise_dim"], device=config["device"]),
            torch.eye(config["noise_dim"], device=config["device"])
        ) #isotropic

def get_uniform(config):
    return tdists.uniform.Uniform(
            torch.zeros(config["noise_dim"], device=config["device"]),
            torch.ones(config["noise_dim"], device=config["device"])
        ) # Uniform on [0,1]

def get_exponential(config):
    return tdists.exponential.Exponential(
            torch.ones(config["noise_dim"], device=config["device"])
        ) # Exponential, rate 1

noise_dists = {
    "gaussian": get_gaussian,
    "uniform": get_uniform,
    "exponential": get_exponential,
}

def get_noise_dist(config):
    dist = config["noise_dist"]
    assert (dist in noise_dists), "Unknown noise distribution: {}".format(dist)
    return noise_dists[dist](config)

