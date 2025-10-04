# utils/device_manager.py

""" Import Library """
# Third-party library imports
import torch


# Lazy import to avoid circular dependency
def get_logger():
    """Lazy import logger to prevent circular import"""
    from utils.logger import get_logger as _get_logger
    return _get_logger()


""" Device Manager (Singleton) """
class DeviceManager:
    """
    Centralized device management using Singleton pattern
    Ensures consistent GPU/CPU usage across the project
    Prevents global side effects from torch.set_default_tensor_type()
    """

    _instance = None  # Singleton instance
    _device = None  # PyTorch device

    def __new__(cls):
        """Singleton pattern: only one instance exists"""
        if cls._instance is None:
            cls._instance = super(DeviceManager, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize device on first instantiation"""
        if self._device is None:
            self._setup_device()

    def _setup_device(self):
        """
        Setup device once at initialization
        Automatically selects CUDA if available, otherwise CPU
        """
        logger = get_logger()
        if torch.cuda.is_available():
            self._device = torch.device("cuda")
            logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            self._device = torch.device("cpu")
            logger.info("CUDA not available. Using CPU")

    @property
    def device(self):
        """Get current device (cuda or cpu)"""
        return self._device


    """ Tensor Creation Methods """
    def tensor(self, data, dtype=torch.float32, requires_grad=False):
        """
        Create tensor on the correct device

        Args:
            data: Data to convert to tensor
            dtype: Data type (default: float32)
            requires_grad: Enable gradient tracking (default: False)

        Returns:
            Tensor on the current device
        """
        return torch.tensor(data, dtype=dtype, device=self._device, requires_grad=requires_grad)

    def zeros(self, *size, dtype=torch.float32, requires_grad=False):
        """
        Create zero tensor on the correct device

        Args:
            size: Tensor dimensions
            dtype: Data type (default: float32)
            requires_grad: Enable gradient tracking (default: False)

        Returns:
            Zero tensor on the current device
        """
        return torch.zeros(*size, dtype=dtype, device=self._device, requires_grad=requires_grad)

    def ones(self, *size, dtype=torch.float32, requires_grad=False):
        """
        Create ones tensor on the correct device

        Args:
            size: Tensor dimensions
            dtype: Data type (default: float32)
            requires_grad: Enable gradient tracking (default: False)

        Returns:
            Ones tensor on the current device
        """
        return torch.ones(*size, dtype=dtype, device=self._device, requires_grad=requires_grad)

    def from_numpy(self, array, dtype=torch.float32):
        """
        Create tensor from numpy array on the correct device

        Args:
            array: Numpy array
            dtype: Data type (default: float32)

        Returns:
            Tensor on the current device
        """
        return torch.from_numpy(array).to(dtype=dtype, device=self._device)

    def to_numpy(self, tensor):
        """
        Convert tensor to numpy array (handles device transfer)

        Args:
            tensor: PyTorch tensor

        Returns:
            Numpy array (on CPU)
        """
        return tensor.detach().cpu().numpy()


""" Global Access Functions """
# Global singleton instance
device_manager = DeviceManager()


def get_device():
    """Get the global device (cuda or cpu)"""
    return device_manager.device


def get_device_manager():
    """Get the global device manager instance"""
    return device_manager

# EOS - End of Script