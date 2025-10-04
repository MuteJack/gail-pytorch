# env/longitudinal_driving_env/longitudinal_driving_env.py

""" Import Library """
# Standard library imports
import numpy as np

# Third-party library imports
import gym
from gym import spaces


class LongitudinalDrivingEnv(gym.Env):
	"""
	Simple Longitudinal Driving Environment

	Input (Action):
		- acceleration: 가속도 (m/s²)

	Output (Observation):
		- velocity: 차량 속도 (m/s)
		- jerk: 저크 (m/s³)
		- rel_dist: 상대 거리 (m)
		- rel_velocity: 상대 속도 (m/s)
		- flag: 상태 플래그 (0=정상, 1=경고)
	"""

	metadata = {'render.modes': ['human']}

	def __init__(self, dt=0.05, max_steps=1000):
		super().__init__()

		# Simulation parameters
		self.dt = dt  # Time step (50ms)
		self.max_steps = max_steps

		# Action space: acceleration (-5 to 5 m/s²)
		self.action_space = spaces.Box(
			low		= 5.0,
			high	= 5.0,
			shape	= (1,),
			dtype	= np.float32
		)

		# Observation space: [velocity, jerk, rel_dist, rel_velocity, flag]
		self.observation_space = spaces.Box(
			#		= np.array(	[	vel, 		jerk, 		rel_dist, 	rel_vel, 	flag] ),
			low		= np.array( [	0.00,		-10.0,		-0.000,		-30.0,		+0.0] ),
			high	= np.array( [	100.0,		+10.0,		+200.0,		+30.0,		+1.0] ),
			dtype	= np.float32
		)

		# State variables
		self.velocity 			= None
		self.prev_acceleration 	= None
		self.jerk 			= None
		self.rel_dist 		= None
		self.rel_velocity 	= None
		self.flag 			= None

		# Lead vehicle (간단한 모델)
		self.lead_velocity 	= None
		self.lead_position 	= None

		# Episode tracking
		self.current_step 	= None
		self.position 		= None

		self.reset()

	def reset(self):
		"""Reset the environment"""
		# Ego vehicle initial state
		self.velocity = np.random.uniform(10.0, 20.0)  # 10-20 m/s
		self.position = 0.0
		self.prev_acceleration = 0.0
		self.jerk = 0.0

		# Lead vehicle initial state
		self.lead_velocity = np.random.uniform(10.0, 20.0)
		self.lead_position = np.random.uniform(30.0, 50.0)  # 30-50m ahead

		# Relative state
		self.rel_dist = self.lead_position - self.position
		self.rel_velocity = self.lead_velocity - self.velocity

		# Flag (0: normal, 1: warning)
		self.flag = 0.0

		self.current_step = 0

		return self._get_obs()

	def step(self, action):
		"""Execute one time step"""
		# Extract acceleration from action
		acceleration = np.clip(action[0], -5.0, 5.0)

		# Calculate jerk (derivative of acceleration)
		self.jerk = (acceleration - self.prev_acceleration) / self.dt
		self.prev_acceleration = acceleration

		# Update ego vehicle state
		self.velocity = np.clip(
			self.velocity + acceleration * self.dt,
			0.0,
			50.0  # Max velocity 50 m/s (~180 km/h)
		)
		self.position += self.velocity * self.dt

		# Update lead vehicle (simple constant velocity model)
		# Add small random perturbation
		lead_accel = np.random.uniform(-0.5, 0.5)
		self.lead_velocity = np.clip(
			self.lead_velocity + lead_accel * self.dt,
			5.0,
			30.0
		)
		self.lead_position += self.lead_velocity * self.dt

		# Update relative state
		self.rel_dist = self.lead_position - self.position
		self.rel_velocity = self.lead_velocity - self.velocity

		# Update flag (warning if too close or closing too fast)
		if self.rel_dist < 10.0 or (self.rel_velocity < -5.0 and self.rel_dist < 30.0):
			self.flag = 1.0
		else:
			self.flag = 0.0

		# Get observation
		obs = self._get_obs()

		# Calculate reward
		reward = self._calculate_reward(acceleration)

		# Check if done
		self.current_step += 1
		done = (
			self.current_step >= self.max_steps or
			self.rel_dist < 2.0 or  # Collision
			self.rel_dist > 150.0  # Lost lead vehicle
		)

		# Info
		info = {
			'position': self.position,
			'lead_position': self.lead_position,
			'collision': self.rel_dist < 2.0
		}

		return obs, reward, done, info

	def _get_obs(self):
		"""Get current observation"""
		return np.array([
			self.velocity,
			self.jerk,
			self.rel_dist,
			self.rel_velocity,
			self.flag
		], dtype=np.float32)

	def _calculate_reward(self, acceleration):
		"""Calculate reward"""
		# Reward components
		# 1. Maintain safe distance (30-50m)
		target_dist = 40.0
		dist_error = abs(self.rel_dist - target_dist)
		dist_reward = -0.1 * dist_error

		# 2. Match lead vehicle velocity
		vel_error = abs(self.rel_velocity)
		vel_reward = -0.5 * vel_error

		# 3. Minimize jerk (comfort)
		jerk_penalty = -0.01 * abs(self.jerk)

		# 4. Collision penalty
		collision_penalty = -100.0 if self.rel_dist < 2.0 else 0.0

		# 5. Warning penalty
		warning_penalty = -1.0 if self.flag == 1.0 else 0.0

		total_reward = (
			dist_reward +
			vel_reward +
			jerk_penalty +
			collision_penalty +
			warning_penalty
		)

		return total_reward

	def render(self, mode='human'):
		"""Render the environment"""
		if mode == 'human':
			print(f"Step: {self.current_step}")
			print(f"Ego - Pos: {self.position:.1f}m, Vel: {self.velocity:.1f}m/s")
			print(f"Lead - Pos: {self.lead_position:.1f}m, Vel: {self.lead_velocity:.1f}m/s")
			print(f"Relative - Dist: {self.rel_dist:.1f}m, Vel: {self.rel_velocity:.1f}m/s")
			print(f"Jerk: {self.jerk:.2f}m/s³, Flag: {int(self.flag)}")
			print("-" * 50)

	def close(self):
		"""Clean up"""
		pass

# EOS - End of Script
