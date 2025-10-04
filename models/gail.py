# models/gail.py

""" Import Library """
# Third-party library imports
import numpy as np
import torch
from torch.nn import Module

# Local application imports
from models.nets import PolicyNetwork, ValueNetwork, Discriminator
from utils.funcs import get_flat_grads, get_flat_params, set_params, \
    conjugate_gradient, rescale_and_linesearch
from utils.device_manager import get_device_manager
from utils.logger import get_logger
from utils.env_logger import EnvStateLogger, EnvInfoLogger

# Initialize logger
logger = get_logger()


""" GAIL Model """
class GAIL(Module):
    """
    Generative Adversarial Imitation Learning (GAIL)
    Uses discriminator to distinguish expert from learner behavior
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
        hidden_dim = train_config.get("hidden_dim", 50)
        num_hidden_layers = train_config.get("hidden_layers", 3)

        
        """ Network Setting (Policy, Discriminator, Value) """
        logger.info("Creating Network....")
        logger.info(f"Network Settings >> discrete: {self.discrete}, state: {self.state_dim}, action={self.action_dim}, layer: {num_hidden_layers}, hidden_dimmension: {hidden_dim}")
        
        # Policy network (Actor)
        self.pi = PolicyNetwork(self.state_dim, self.action_dim, self.discrete, hidden_dim, num_hidden_layers)
        
        # Value network (Critic) - for GAE
        self.v = ValueNetwork(self.state_dim, hidden_dim, num_hidden_layers)

        # Discriminator - distinguishes expert from learner
        self.d = Discriminator(self.state_dim, self.action_dim, self.discrete, hidden_dim, num_hidden_layers)

    def get_networks(self):
        """Return policy and value networks (not discriminator)"""
        return [self.pi, self.v]

    def act(self, state):
        """
        Select action from policy network

        Args:
            state: Current environment state

        Returns:
            action: Sampled action from policy distribution
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

    def train(self, env, expert, render=False):
        """
        Train GAIL model using expert demonstrations

        Args:
            env: Gym environment
            expert: Expert model to collect demonstrations
            render: Render environment (default: False)

        Returns:
            exp_rwd_mean: Expert average reward
            rwd_iter_means: Learner average rewards per iteration
        """

        """ Training Configuration """
        # Extract hyperparameters from config
        num_iters = self.train_config["num_iters"]  # Number of training iterations
        num_steps_per_iter = self.train_config["num_steps_per_iter"]  # Steps per iteration
        horizon = self.train_config["horizon"]  # Max steps per episode
        lambda_ = self.train_config["lambda"]  # Entropy regularization coefficient
        gae_gamma = self.train_config["gae_gamma"]  # GAE discount factor
        gae_lambda = self.train_config["gae_lambda"]  # GAE lambda parameter
        eps = self.train_config["epsilon"]  # Value network constraint
        max_kl = self.train_config["max_kl"]  # Max KL divergence for policy update
        cg_damping = self.train_config["cg_damping"]  # Conjugate gradient damping
        normalize_advantage = self.train_config["normalize_advantage"]  # Normalize advantages

        # Discriminator optimizer
        opt_d = torch.optim.Adam(self.d.parameters())

        # Device manager for tensor operations
        dm = get_device_manager()


        """ Environment Logger Setup """
        # Get environment name from env object
        env_name = getattr(env.spec, 'id', 'unknown') if hasattr(env, 'spec') and env.spec else 'unknown'

        # Initialize environment loggers (always enabled, shared timestamp)
        from datetime import datetime
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        state_logger = EnvStateLogger(env_name=env_name, timestamp=timestamp)
        info_logger = EnvInfoLogger(env_name=env_name, timestamp=timestamp)


        """ Collect Expert Demonstrations """
        exp_rwd_iter = []  # Expert rewards per episode

        exp_obs = []  # Expert observations
        exp_acts = []  # Expert actions

        episode_count = 0
        steps = 0
        while steps < num_steps_per_iter:
            # Episode-level storage
            ep_obs = []
            ep_rwds = []

            t = 0  # Episode timestep
            done = False

            # Reset environment
            ob = env.reset()

            # Run episode
            while not done and steps < num_steps_per_iter:
                # Expert takes action
                act = expert.act(ob)

                # Store observation and action
                ep_obs.append(ob)
                exp_obs.append(ob)
                exp_acts.append(act)

                # Render if requested
                if render:
                    env.render()

                # Environment step
                ob, rwd, done, info = env.step(act)

                # Log environment state and info
                state_logger.log_step(ep_obs[-1], act, rwd, done)
                info_logger.log_step(info, episode_count, steps)

                ep_rwds.append(rwd)

                t += 1
                steps += 1

                # Check horizon limit
                if horizon is not None:
                    if t >= horizon:
                        done = True
                        break

            # Episode finished
            if done:
                exp_rwd_iter.append(np.sum(ep_rwds))
                state_logger.new_episode()
                episode_count += 1

            # Convert to tensors
            ep_obs = dm.from_numpy(np.array(ep_obs))
            ep_rwds = dm.tensor(ep_rwds)

        # Calculate expert performance
        exp_rwd_mean = np.mean(exp_rwd_iter)
        logger.info(f"Expert Reward Mean: {exp_rwd_mean:.4f}")

        # Convert expert data to tensors
        exp_obs = dm.from_numpy(np.array(exp_obs))
        exp_acts = dm.from_numpy(np.array(exp_acts))

        """ GAIL Training Loop """
        rwd_iter_means = []
        for i in range(num_iters):
            rwd_iter = []  # Rewards for this iteration

            # Batch-level storage
            obs = []  # All observations
            acts = []  # All actions
            rets = []  # Returns
            advs = []  # Advantages
            gms = []  # Gamma values

            steps = 0
            while steps < num_steps_per_iter:
                # Episode-level storage
                ep_obs = []
                ep_acts = []
                ep_rwds = []
                ep_costs = []  # Discriminator costs
                ep_disc_costs = []  # Discounted costs
                ep_gms = []  # Gammas for this episode
                ep_lmbs = []  # Lambdas for this episode

                t = 0  # Episode timestep
                done = False

                # Reset environment
                ob = env.reset()

                # Run episode with learner policy
                while not done and steps < num_steps_per_iter:
                    # Learner takes action
                    act = self.act(ob)

                    # Store observation and action
                    ep_obs.append(ob)
                    obs.append(ob)

                    ep_acts.append(act)
                    acts.append(act)

                    # Render if requested
                    if render:
                        env.render()

                    # Environment step
                    ob, rwd, done, info = env.step(act)

                    # Store rewards and discount factors
                    ep_rwds.append(rwd)
                    ep_gms.append(gae_gamma ** t)  # Discount factor
                    ep_lmbs.append(gae_lambda ** t)  # Lambda for GAE

                    t += 1
                    steps += 1

                    # Check horizon limit
                    if horizon is not None:
                        if t >= horizon:
                            done = True
                            break

                # Episode finished
                if done:
                    rwd_iter.append(np.sum(ep_rwds))

                # Convert episode data to tensors
                ep_obs = dm.from_numpy(np.array(ep_obs))
                ep_acts = dm.from_numpy(np.array(ep_acts))
                ep_rwds = dm.tensor(ep_rwds)
                ep_gms = dm.tensor(ep_gms)
                ep_lmbs = dm.tensor(ep_lmbs)

                # Compute discriminator costs (GAIL reward)
                ep_costs = (-1) * torch.log(self.d(ep_obs, ep_acts))\
                    .squeeze().detach()
                ep_disc_costs = ep_gms * ep_costs

                # Compute returns from discriminator costs
                ep_disc_rets = dm.tensor(
                    [sum(ep_disc_costs[i:]) for i in range(t)]
                )
                ep_rets = ep_disc_rets / ep_gms

                rets.append(ep_rets)

                # Compute advantages using GAE
                self.v.eval()
                curr_vals = self.v(ep_obs).detach()  # V(s_t)
                next_vals = torch.cat(
                    (self.v(ep_obs)[1:], dm.zeros(1, 1))
                ).detach()  # V(s_{t+1})
                ep_deltas = ep_costs.unsqueeze(-1)\
                    + gae_gamma * next_vals\
                    - curr_vals  # TD error

                # GAE computation
                ep_advs = dm.tensor([
                    ((ep_gms * ep_lmbs)[:t - j].unsqueeze(-1) * ep_deltas[j:])
                    .sum()
                    for j in range(t)
                ])
                advs.append(ep_advs)

                gms.append(ep_gms)

            # Log iteration performance
            rwd_iter_means.append(np.mean(rwd_iter))
            logger.info(f"Iteration {i + 1}/{num_iters} - Reward Mean: {np.mean(rwd_iter):.4f}")


            """ Prepare Batch Data """
            # Convert lists to tensors
            obs = dm.from_numpy(np.array(obs))
            acts = dm.from_numpy(np.array(acts))
            rets = torch.cat(rets)
            advs = torch.cat(advs)
            gms = torch.cat(gms)

            # Normalize advantages
            if normalize_advantage:
                advs = (advs - advs.mean()) / advs.std()


            """ Update Discriminator """
            self.d.train()
            exp_scores = self.d.get_logits(exp_obs, exp_acts)  # Expert scores
            nov_scores = self.d.get_logits(obs, acts)  # Learner scores

            # Binary cross-entropy loss (expert=0, learner=1)
            opt_d.zero_grad()
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                exp_scores, torch.zeros_like(exp_scores)
            ) \
                + torch.nn.functional.binary_cross_entropy_with_logits(
                    nov_scores, torch.ones_like(nov_scores)
                )
            loss.backward()
            opt_d.step()


            """ Update Value Network (TRPO-style) """
            self.v.train()
            old_params = get_flat_params(self.v).detach()
            old_v = self.v(obs).detach()

            # KL divergence constraint for value network
            def constraint():
                return ((old_v - self.v(obs)) ** 2).mean()

            grad_diff = get_flat_grads(constraint(), self.v)

            # Hessian-vector product
            def Hv(v):
                hessian = get_flat_grads(torch.dot(grad_diff, v), self.v)\
                    .detach()
                return hessian

            # Value loss gradient
            g = get_flat_grads(
                ((-1) * (self.v(obs).squeeze() - rets) ** 2).mean(), self.v
            ).detach()

            # Conjugate gradient to solve Hv = g
            s = conjugate_gradient(Hv, g).detach()

            # Compute step size
            Hs = Hv(s).detach()
            alpha = torch.sqrt(2 * eps / torch.dot(s, Hs))

            # Update value network parameters
            new_params = old_params + alpha * s
            set_params(self.v, new_params)


            """ Update Policy Network (TRPO) """
            self.pi.train()
            old_params = get_flat_params(self.pi).detach()
            old_distb = self.pi(obs)

            # Surrogate loss (policy improvement objective)
            def L():
                distb = self.pi(obs)
                return (advs * torch.exp(
                            distb.log_prob(acts)
                            - old_distb.log_prob(acts).detach()
                        )).mean()

            # KL divergence constraint
            def kld():
                distb = self.pi(obs)

                if self.discrete:
                    # Discrete action space
                    old_p = old_distb.probs.detach()
                    p = distb.probs
                    return (old_p * (torch.log(old_p) - torch.log(p)))\
                        .sum(-1)\
                        .mean()
                else:
                    # Continuous action space
                    old_mean = old_distb.mean.detach()
                    old_cov = old_distb.covariance_matrix.sum(-1).detach()
                    mean = distb.mean
                    cov = distb.covariance_matrix.sum(-1)

                    return (0.5) * (
                            (old_cov / cov).sum(-1)
                            + (((old_mean - mean) ** 2) / cov).sum(-1)
                            - self.action_dim
                            + torch.log(cov).sum(-1)
                            - torch.log(old_cov).sum(-1)
                        ).mean()

            # KL divergence gradient
            grad_kld_old_param = get_flat_grads(kld(), self.pi)

            # Hessian-vector product with damping
            def Hv(v):
                hessian = get_flat_grads(
                    torch.dot(grad_kld_old_param, v),
                    self.pi
                ).detach()
                return hessian + cg_damping * v

            # Policy loss gradient
            g = get_flat_grads(L(), self.pi).detach()

            # Conjugate gradient to solve Hv = g
            s = conjugate_gradient(Hv, g).detach()
            Hs = Hv(s).detach()

            # Line search to satisfy KL constraint
            new_params = rescale_and_linesearch(
                g, s, Hs, max_kl, L, kld, old_params, self.pi
            )

            # Add entropy regularization
            disc_causal_entropy = ((-1) * gms * self.pi(obs).log_prob(acts))\
                .mean()
            grad_disc_causal_entropy = get_flat_grads(
                disc_causal_entropy, self.pi
            )
            new_params += lambda_ * grad_disc_causal_entropy

            # Update policy network parameters
            set_params(self.pi, new_params)


        """ End of Training """
        # Close environment loggers
        state_logger.close()
        info_logger.close()

        return exp_rwd_mean, rwd_iter_means

# EOS - End of Script