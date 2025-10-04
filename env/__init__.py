# env/__init__.py

""" Custom Gym Environments Registration """
from gym.envs.registration import register

# Longitudinal Driving Environment
register(
    id='LongitudinalDriving-v0',
    entry_point='env.longitudinal_driving_env.longitudinal_driving_env:LongitudinalDrivingEnv',
    max_episode_steps=1000,
)

__all__ = ['LongitudinalDrivingEnv']
