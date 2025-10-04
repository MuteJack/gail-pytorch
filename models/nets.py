# models/nets.py

""" Import Library """
# Third-party library imports
import torch
from torch.nn import Module, Sequential, Linear, Tanh, Parameter, Embedding
from torch.distributions import Categorical, MultivariateNormal

# Local application imports
from utils.device_manager import get_device_manager


""" MLP Builder Function """
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

    # Input layer: input_dim -> hidden_dim
    layers.append(Linear(input_dim, hidden_dim))
    layers.append(Tanh())

    # Hidden layers: hidden_dim -> hidden_dim (repeated)
    for _ in range(num_hidden_layers - 1):
        layers.append(Linear(hidden_dim, hidden_dim))
        layers.append(Tanh())

    # Output layer: hidden_dim -> output_dim (no activation)
    layers.append(Linear(hidden_dim, output_dim))

    return Sequential(*layers)



""" Policy Network (Actor) """
class PolicyNetwork(Module):
    """
    Policy network that outputs action distribution
    Supports both discrete and continuous action spaces
    """
    def __init__(self, state_dim, action_dim, discrete, hidden_dim=50, num_hidden_layers=3) -> None:
        super().__init__()

        # MLP for policy
        self.net = create_mlp(state_dim, action_dim, hidden_dim, num_hidden_layers)

        # Environment dimensions
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.discrete = discrete

        # Learnable log standard deviation for continuous actions
        if not self.discrete:
            self.log_std = Parameter(torch.zeros(action_dim))

    def forward(self, states):
        """
        Forward pass to get action distribution

        Args:
            states: Batch of states

        Returns:
            distb: Action distribution (Categorical or MultivariateNormal)
        """
        if self.discrete:
            # Discrete action space: output probabilities
            probs = torch.softmax(self.net(states), dim=-1)
            distb = Categorical(probs)
        else:
            # Continuous action space: output mean and std
            mean = self.net(states)

            std = torch.exp(self.log_std)
            cov_mtx = torch.eye(self.action_dim, device=std.device) * (std ** 2)

            distb = MultivariateNormal(mean, cov_mtx)

        return distb



""" Value Network (Critic) """
class ValueNetwork(Module):
    """
    Value network that estimates state value V(s)
    Used for GAE (Generalized Advantage Estimation)
    """
    def __init__(self, state_dim, hidden_dim=50, num_hidden_layers=3) -> None:
        super().__init__()

        # MLP: state -> value (scalar)
        self.net = create_mlp(state_dim, 1, hidden_dim, num_hidden_layers)

    def forward(self, states):
        """
        Forward pass to estimate value

        Args:
            states: Batch of states

        Returns:
            values: State values V(s)
        """
        return self.net(states)



""" Discriminator Network """
class Discriminator(Module):
    """
    Discriminator network for GAIL
    Distinguishes expert trajectories from learner trajectories
    """
    def __init__(self, state_dim, action_dim, discrete, hidden_dim=50, num_hidden_layers=3) -> None:
        super().__init__()

        # Environment dimensions
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.discrete = discrete

        # Input dimension depends on action type
        if self.discrete:
            # Discrete: embed action then concatenate with state
            self.act_emb = Embedding(action_dim, state_dim)
            self.net_in_dim = 2 * state_dim
        else:
            # Continuous: concatenate state and action directly
            self.net_in_dim = state_dim + action_dim

        # MLP: (state, action) -> logit (scalar)
        self.net = create_mlp(self.net_in_dim, 1, hidden_dim, num_hidden_layers)

    def forward(self, states, actions):
        """
        Forward pass to get probability

        Args:
            states: Batch of states
            actions: Batch of actions

        Returns:
            probs: Probability of being learner trajectory
        """
        return torch.sigmoid(self.get_logits(states, actions))

    def get_logits(self, states, actions):
        """
        Get raw logits before sigmoid

        Args:
            states: Batch of states
            actions: Batch of actions

        Returns:
            logits: Raw discriminator outputs
        """
        if self.discrete:
            # Embed discrete actions
            actions = self.act_emb(actions.long())

        # Concatenate state and action
        sa = torch.cat([states, actions], dim=-1)

        return self.net(sa)


""" Expert Model """
class Expert(Module):
    """
    Expert model with pre-trained policy
    Used to generate expert demonstrations for GAIL
    """
    def __init__(
        self,
        state_dim,
        action_dim,
        discrete,
        train_config=None
    ) -> None:
        super().__init__()

        # Environment dimensions
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.discrete = discrete
        self.train_config = train_config

        # Extract network architecture parameters
        if train_config:
            hidden_dim = train_config.get("hidden_dim", 50)
            num_hidden_layers = train_config.get("hidden_layers", 3)
        else:
            hidden_dim = 50
            num_hidden_layers = 3

        # Policy network only (no value or discriminator)
        self.pi = PolicyNetwork(self.state_dim, self.action_dim, self.discrete,
                                hidden_dim, num_hidden_layers)

    def get_networks(self):
        """Return list of networks"""
        return [self.pi]

    def act(self, state):
        """
        Select action from expert policy

        Args:
            state: Current environment state

        Returns:
            action: Sampled action from expert policy
        """
        self.pi.eval()

        # Convert numpy state to tensor
        dm = get_device_manager()
        state = dm.from_numpy(state)

        # Get action distribution
        distb = self.pi(state)

        # Sample action and convert to numpy
        action = dm.to_numpy(distb.sample())

        return action

# EOS - End of Script
