# utils/device_manager.py

""" Import Library """
# Third-party library imports
import torch


# Lazy import to avoid circular dependency
def get_logger():
    from utils.logger import get_logger as _get_logger
    return _get_logger()


class DeviceManager:
    """Centralized device management for the entire project"""

    _instance = None
    _device = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DeviceManager, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if self._device is None:
            self._setup_device()

    def _setup_device(self):
        """Setup device once at initialization"""
        logger = get_logger()
        if torch.cuda.is_available():
            self._device = torch.device("cuda")
            logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            self._device = torch.device("cpu")
            logger.info("CUDA not available. Using CPU")

    @property
    def device(self):
        """Get current device"""
        return self._device

    def tensor(self, data, dtype=torch.float32, requires_grad=False):
        """Create tensor on the correct device"""
        return torch.tensor(data, dtype=dtype, device=self._device, requires_grad=requires_grad)

    def zeros(self, *size, dtype=torch.float32, requires_grad=False):
        """Create zero tensor on the correct device"""
        return torch.zeros(*size, dtype=dtype, device=self._device, requires_grad=requires_grad)

    def ones(self, *size, dtype=torch.float32, requires_grad=False):
        """Create ones tensor on the correct device"""
        return torch.ones(*size, dtype=dtype, device=self._device, requires_grad=requires_grad)

    def from_numpy(self, array, dtype=torch.float32):
        """Create tensor from numpy array on the correct device"""
        return torch.from_numpy(array).to(dtype=dtype, device=self._device)

    def to_numpy(self, tensor):
        """Convert tensor to numpy array (handles device transfer)"""
        return tensor.detach().cpu().numpy()


# Global singleton instance
device_manager = DeviceManager()


def get_device():
    """Get the global device"""
    return device_manager.device


def get_device_manager():
    """Get the global device manager instance"""
    return device_manager