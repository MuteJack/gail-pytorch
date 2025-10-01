# models/nets.py

""" Import Library """
# Third-party library imports
import torch
from torch.nn import Module, Sequential, Linear, Tanh, Parameter, Embedding
from torch.distributions import Categorical, MultivariateNormal

# Local application imports
from utils.device_manager import get_device_manager


def create_mlp(input_dim, output_dim, hidden_dim=50, num_hidden_layers=3):
    """
    Create a Multi-Layer Perceptron (MLP)

    Args:
        input_dim: Input dimension
        output_dim: Output dimension
        hidden_dim: Hidden layer dimension (default: 50)
        num_hidden_layers: Number of hidden layers (default: 3)

    Returns:
        Sequential module containing the MLP layers

    Example:
        >>> create_mlp(state_dim, action_dim, hidden_dim=50, num_hidden_layers=3)
        Sequential(
            Linear(state_dim, 50),
            Tanh(),
            Linear(50, 50),
            Tanh(),
            Linear(50, 50),
            Tanh(),
            Linear(50, action_dim)
        )
    """
    layers = []

    # Input layer
    layers.append(Linear(input_dim, hidden_dim))
    layers.append(Tanh())

    # Hidden layers
    for _ in range(num_hidden_layers - 1):
        layers.append(Linear(hidden_dim, hidden_dim))
        layers.append(Tanh())

    # Output layer
    layers.append(Linear(hidden_dim, output_dim))

    return Sequential(*layers)



""" Policy pi """
class PolicyNetwork(Module):
    def __init__(self, state_dim, action_dim, discrete, hidden_dim=50, num_hidden_layers=3) -> None:
        super().__init__()

        self.net = create_mlp(state_dim, action_dim, hidden_dim, num_hidden_layers)

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.discrete = discrete

        if not self.discrete:
            self.log_std = Parameter(torch.zeros(action_dim))

    def forward(self, states):
        if self.discrete:
            probs = torch.softmax(self.net(states), dim=-1)
            distb = Categorical(probs)
        else:
            mean = self.net(states)

            std = torch.exp(self.log_std)
            cov_mtx = torch.eye(self.action_dim) * (std ** 2)

            distb = MultivariateNormal(mean, cov_mtx)

        return distb



""" Value V """
class ValueNetwork(Module):
    def __init__(self, state_dim, hidden_dim=50, num_hidden_layers=3) -> None:
        super().__init__()

        self.net = create_mlp(state_dim, 1, hidden_dim, num_hidden_layers)

    def forward(self, states):
        return self.net(states)



""" Discriminator D """
class Discriminator(Module):
    def __init__(self, state_dim, action_dim, discrete, hidden_dim=50, num_hidden_layers=3) -> None:
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.discrete = discrete

        if self.discrete:
            self.act_emb = Embedding(action_dim, state_dim)
            self.net_in_dim = 2 * state_dim
        else:
            self.net_in_dim = state_dim + action_dim

        self.net = create_mlp(self.net_in_dim, 1, hidden_dim, num_hidden_layers)

    def forward(self, states, actions):
        return torch.sigmoid(self.get_logits(states, actions))

    def get_logits(self, states, actions):
        if self.discrete:
            actions = self.act_emb(actions.long())

        sa = torch.cat([states, actions], dim=-1)

        return self.net(sa)


class Expert(Module):
    def __init__(
        self,
        state_dim,
        action_dim,
        discrete,
        train_config=None
    ) -> None:
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.discrete = discrete
        self.train_config = train_config

        self.pi = PolicyNetwork(self.state_dim, self.action_dim, self.discrete)

    def get_networks(self):
        return [self.pi]

    def act(self, state):
        self.pi.eval()

        dm = get_device_manager()
        state = dm.from_numpy(state)
        distb = self.pi(state)

        action = dm.to_numpy(distb.sample())

        return action
