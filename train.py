# train.py

""" Import Library """
# Standard library imports
import os
import json
import pickle
import argparse

# Third-party library imports
import torch
import gym

# Local application imports
from models.nets import Expert
from models.gail import GAIL
from utils.device_manager import get_device
from utils.logger import get_logger

logger = get_logger()

""" Global Variable """
ENV_LIST = ["CartPole-v1", "Pendulum-v0", "BipedalWalker-v3"]

def main(env_name):
    # Directories
    CKPT_DIR = "ckpts"      # Base Directory where the CheckPoint will be saved
    EXPERT_DIR = "experts"  # Base Directory where the CheckPoint of Expert is located

    """ Directory SetUp """
    # Check Point Directory
    if not os.path.isdir(CKPT_DIR): os.mkdir(CKPT_DIR)

    ckpt_path = os.path.join(CKPT_DIR, env_name)
    if not os.path.isdir(ckpt_path):
        os.mkdir(ckpt_path)
    with open("config.json") as f:
        config = json.load(f)[env_name]
    with open(os.path.join(ckpt_path, "model_config.json"), "w") as f:
        json.dump(config, f, indent=4)

    # Expert Check Point Directory
    expert_path = os.path.join(EXPERT_DIR, env_name)
    with open(os.path.join(expert_path, "model_config.json")) as f:
        expert_config = json.load(f)



    """ Environment SetUp """
    # Check Environment is in the List
    if env_name not in ENV_LIST:
        logger.error(f"Invalid environment name: {env_name}")
        logger.info(f"Available environments: {ENV_LIST}")
        return

    # Make GYM Environment
    env = gym.make(env_name)
    env.reset()

    state_dim = len(env.observation_space.high)
    if env_name in ["CartPole-v1"]:
        discrete = True
        action_dim = env.action_space.n
    else:
        discrete = False
        action_dim = env.action_space.shape[0]



    """ Model SetUp """
    # Get device from centralized device manager
    device = get_device()



    """ Expert (Model) SetUp """
    expert = Expert(
        state_dim, action_dim, discrete, **expert_config
    ).to(device)
    expert.pi.load_state_dict(
        torch.load(
            os.path.join(expert_path, "policy.ckpt"), map_location=device
        )
    )



    """ Training Model SetUp """
    model = GAIL(state_dim, action_dim, discrete, config).to(device)
    results = model.train(env, expert)
    env.close()



    """ End of Training """
    with open(os.path.join(ckpt_path, "results.pkl"), "wb") as f:
        pickle.dump(results, f)

    if hasattr(model, "pi"):
        torch.save(  model.pi.state_dict(), os.path.join(ckpt_path, "policy.ckpt")  )
    if hasattr(model, "v"):
        torch.save(  model.v.state_dict(), os.path.join(ckpt_path, "value.ckpt")  )
    if hasattr(model, "d"):
        torch.save(  model.d.state_dict(), os.path.join(ckpt_path, "discriminator.ckpt")  )

# EOS - End of Script


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env_name",
        type=str,
        default="CartPole-v1",
        help="Type the environment name to run. \
            The possible environments are \
                [CartPole-v1, Pendulum-v0, BipedalWalker-v3]"
    )
    args = parser.parse_args()

    main(**vars(args))

# EOF - End of File
