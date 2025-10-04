# test_env.py

""" Test Custom Environment """
import gym
import numpy as np
import env  # Register custom environments

def test_longitudinal_driving():
    """Test LongitudinalDriving-v0 environment"""
    print("Testing LongitudinalDriving-v0...")

    env_instance = gym.make('LongitudinalDriving-v0')

    print(f"\nAction space: {env_instance.action_space}")
    print(f"Observation space: {env_instance.observation_space}")

    # Test reset
    obs = env_instance.reset()
    print(f"\nInitial observation: {obs}")
    print(f"  - velocity: {obs[0]:.2f} m/s")
    print(f"  - jerk: {obs[1]:.2f} m/s³")
    print(f"  - rel_dist: {obs[2]:.2f} m")
    print(f"  - rel_velocity: {obs[3]:.2f} m/s")
    print(f"  - flag: {int(obs[4])}")

    # Test a few steps
    print("\nRunning 10 random steps...")
    for i in range(10):
        action = env_instance.action_space.sample()
        obs, reward, done, info = env_instance.step(action)

        print(f"\nStep {i+1}:")
        print(f"  Action (accel): {action[0]:.2f} m/s²")
        print(f"  Observation: vel={obs[0]:.1f}, jerk={obs[1]:.1f}, rel_dist={obs[2]:.1f}, rel_vel={obs[3]:.1f}, flag={int(obs[4])}")
        print(f"  Reward: {reward:.2f}")
        print(f"  Done: {done}")

        if done:
            print(f"  Episode finished! Collision: {info.get('collision', False)}")
            break

    env_instance.close()
    print("\n✅ Test completed successfully!")

if __name__ == "__main__":
    test_longitudinal_driving()
