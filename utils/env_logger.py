# utils/env_logger.py

""" Import Library """
# Standard library imports
import os
import csv
from datetime import datetime

# Third-party library imports
import numpy as np


""" Environment State Logger """
class EnvStateLogger:
    """
    CSV logger for environment states during training
    Logs state, action, reward, done (not info)
    """

    def __init__(self, log_dir="logs", env_name="unknown", timestamp=None):
        """
        Initialize environment state logger

        Args:
            log_dir: Directory to save CSV logs (default: "logs")
            env_name: Environment name for log filename
            timestamp: Shared timestamp for consistent naming (optional)
        """
        self.log_dir = log_dir
        self.env_name = env_name

        # Create directory if not exists
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        # Create timestamped filename: log_{timestamp}_env_{env_name}_state.csv
        if timestamp is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.timestamp = timestamp
        self.csv_file = os.path.join(log_dir, f'log_{timestamp}_env_{env_name}_state.csv')

        # Initialize variables
        self.csv_writer = None
        self.file_handle = None
        self.episode_count = 0
        self.step_count = 0
        self.header_written = False

    def _get_header(self, state_dim, action_dim):
        """Generate CSV header based on dimensions"""
        # Real-time timestamp first column
        header = ['real_time', 'episode', 'step', 'timestep']

        # State columns
        for i in range(state_dim):
            header.append(f'state_{i}')

        # Action columns
        if isinstance(action_dim, int):
            if action_dim == 1:
                header.append('action')
            else:
                for i in range(action_dim):
                    header.append(f'action_{i}')
        else:
            header.append('action')

        # Additional info
        header.extend(['reward', 'done'])

        return header

    def log_step(self, state, action, reward, done, info=None):
        """
        Log a single step

        Args:
            state: Environment state (observation)
            action: Action taken
            reward: Reward received
            done: Episode done flag
            info: Additional info dict (optional)
        """
        # Open file on first log
        if self.file_handle is None:
            self.file_handle = open(self.csv_file, 'w', newline='', encoding='utf-8')
            self.csv_writer = csv.writer(self.file_handle)

        # Convert to numpy arrays
        state = np.array(state).flatten()
        action = np.array(action).flatten() if hasattr(action, '__iter__') else np.array([action])

        # Write header if first time
        if not self.header_written:
            header = self._get_header(len(state), len(action))
            self.csv_writer.writerow(header)
            self.header_written = True

        # Get current real time (YYYY-MM-DD HH:MM:SS)
        real_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # Prepare row data (real_time first)
        row = [real_time, self.episode_count, self.step_count, self.step_count]
        row.extend(state.tolist())
        row.extend(action.tolist())
        row.extend([reward, int(done)])

        # Write to CSV
        self.csv_writer.writerow(row)

        # Update counters
        self.step_count += 1

        if done:
            self.episode_count += 1
            self.file_handle.flush()  # Ensure data is written

    def new_episode(self):
        """Mark start of new episode"""
        self.step_count = 0

    def close(self):
        """Close the CSV file"""
        if self.file_handle is not None:
            self.file_handle.close()
            print(f"Environment state log saved to: {self.csv_file}")

    def __del__(self):
        """Ensure file is closed on deletion"""
        self.close()



""" Environment Info Logger """
class EnvInfoLogger:
    """
    CSV logger for environment info dict during training
    Logs info returned from env.step()
    """

    def __init__(self, log_dir="logs", env_name="unknown", timestamp=None):
        """
        Initialize environment info logger

        Args:
            log_dir: Directory to save CSV logs (default: "logs")
            env_name: Environment name for log filename
            timestamp: Shared timestamp for consistent naming (optional)
        """
        self.log_dir = log_dir
        self.env_name = env_name

        # Create directory if not exists
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        # Create timestamped filename: log_{timestamp}_env_{env_name}_info.csv
        if timestamp is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.timestamp = timestamp
        self.csv_file = os.path.join(log_dir, f'log_{timestamp}_env_{env_name}_info.csv')

        # Initialize variables
        self.csv_writer = None
        self.file_handle = None
        self.episode_count = 0
        self.step_count = 0
        self.header_written = False
        self.info_keys = None

    def _flatten_dict(self, d, parent_key='', sep='_'):
        """Flatten nested dictionary"""
        items = []
        for k, v in d.items():
            new_key = f'{parent_key}{sep}{k}' if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)

    def log_step(self, info, episode, step):
        """
        Log info dict from env.step()

        Args:
            info: Info dictionary from environment
            episode: Current episode number
            step: Current step number
        """
        # Skip if info is empty
        if not info or not isinstance(info, dict):
            return

        # Open file on first log
        if self.file_handle is None:
            self.file_handle = open(self.csv_file, 'w', newline='', encoding='utf-8')
            self.csv_writer = csv.writer(self.file_handle)

        # Flatten nested dict
        flat_info = self._flatten_dict(info)

        # Write header if first time
        if not self.header_written:
            self.info_keys = list(flat_info.keys())
            header = ['real_time', 'episode', 'step', 'timestep'] + self.info_keys
            self.csv_writer.writerow(header)
            self.header_written = True

        # Get current real time
        real_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # Prepare row data (use consistent key order)
        row = [real_time, episode, step, step]
        for key in self.info_keys:
            row.append(flat_info.get(key, ''))  # Empty string if key missing

        # Write to CSV
        self.csv_writer.writerow(row)

    def close(self):
        """Close the CSV file"""
        if self.file_handle is not None:
            self.file_handle.close()
            print(f"Environment info log saved to: {self.csv_file}")

    def __del__(self):
        """Ensure file is closed on deletion"""
        self.close()

# EOS - End of Script