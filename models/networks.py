import torch.nn as nn
import torch.nn.functional as F
import torch

# Base neural networks used

# Create feedforward network section
def create_ff_section(in_dim, layer_list, extra_from_dim=0):
    if not layer_list:
        # No hidden layers in section
        layers = []
        out_dim = in_dim + extra_from_dim
    else:
        layers = [nn.Linear(in_dim+extra_from_dim, layer_list[0])] +\
            [nn.Linear(from_dim+extra_from_dim, to_dim) for from_dim, to_dim in
                    zip(layer_list[:-1], layer_list[1:])]
        out_dim = layer_list[-1]

    # ModuleList needs to be used for layers to register correctly
    # (for optimizers etc.)
    return nn.ModuleList(layers), out_dim

# Pure in_dim -> out_dim NN
class FeedForward(nn.Module):
    def __init__(self, spec):
        super().__init__()

        self.activation = activations[spec["activation"]]
        self.layer_list, hidden_dim = create_ff_section(
                spec["in_dim"], spec["hidden_layers"])
        self.output_layer = nn.Linear(hidden_dim, spec["out_dim"])

    def forward(self, x):
        for layer in self.layer_list:
            x = self.activation(layer(x))

        # Return logits
        return self.output_layer(x)


# Feed forward network with initial layers where x and other input are handled
# separately. Their representations are then concatenated and passed through joint
# feed forward hidden layers
class DoubleInputNetwork(nn.Module):
    # Other is either y or noise
    def __init__(self, spec):
        super().__init__()

        self.activation = activations[spec["activation"]]
        self.x_dim = spec["x_dim"] # Store for forward prop

        # layers
        self.x_layers, x_repr_dim = create_ff_section(spec["x_dim"],
                spec["x_layers"])
        self.other_layers, other_repr_dim = create_ff_section(
                spec["other_dim"], spec["other_layers"])
        self.hidden_layers, hidden_dim = create_ff_section(
                (x_repr_dim + other_repr_dim), spec["hidden_layers"])
        self.output_layer = nn.Linear(hidden_dim, spec["out_dim"])

    def forward(self, x):
        x_repr = x[:, :self.x_dim]
        other_repr = x[:, self.x_dim:]

        # x representation
        for layer in self.x_layers:
            x_repr = self.activation(layer(x_repr))

        # other representation
        for layer in self.other_layers:
            other_repr = self.activation(layer(other_repr))

        # hidden layers
        hidden_repr = torch.cat((x_repr, other_repr), dim=1)
        for layer in self.hidden_layers:
            hidden_repr = self.activation(layer(hidden_repr))

        # output layer
        output = self.output_layer(hidden_repr) # No activation
        return output

# Feedforward network where the same noise is injected at each layer
class NoiseInjectionNetwork(nn.Module):
    # Other is either y or noise
    def __init__(self, spec):
        super().__init__()

        self.activation = activations[spec["activation"]]
        self.x_dim = spec["x_dim"] # Store for forward prop

        noise_dim = spec["other_dim"]

        # layers
        self.hidden_layers, hidden_dim = create_ff_section(self.x_dim,
                spec["hidden_layers"], extra_from_dim = noise_dim)
        self.output_layer = nn.Linear((hidden_dim + noise_dim), spec["out_dim"])

    def forward(self, x):
        hidden_repr = x[:, :self.x_dim]
        noise = x[:, self.x_dim:]

        # hidden layers
        for layer in self.hidden_layers:
            next_input = torch.cat((hidden_repr, noise), dim=1)
            hidden_repr = self.activation(layer(next_input))

        # output layer
        next_input = torch.cat((hidden_repr, noise), dim=1)
        output = self.output_layer(next_input) # No activation
        return output

activations = {
    "relu": F.relu,
    "leaky_relu": F.leaky_relu,
    "elu": F.elu,
}

networks = {
    "double_track": DoubleInputNetwork,
    "noise_injection": NoiseInjectionNetwork,
    "ff": FeedForward,
}

optimizers = {
    "adam": torch.optim.Adam,
    "sgd": torch.optim.SGD,
    "rmsprop": torch.optim.RMSprop,
    "adagrad": torch.optim.Adagrad,
}

def build_network(network_spec):
    return networks[network_spec["type"]](network_spec)

