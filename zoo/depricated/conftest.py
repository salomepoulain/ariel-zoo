"""
Pytest configuration for torch tests.

Allows testing on different devices via command line:
    pytest --device=cpu   (default)
    pytest --device=cuda  (GPU)
    pytest --device=mps   (Apple Silicon)
"""

import pytest
import torch


def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--device",
        action="store",
        default="cuda",
        help="Device to run tests on: cpu, cuda, or mps",
    )


@pytest.fixture(scope="session")
def test_device(request):
    """Get test device from command line option."""
    device_str = request.config.getoption("--device")

    # Validate device availability
    if device_str == "cuda":
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
    elif device_str == "mps":
        if not torch.backends.mps.is_available():
            pytest.skip("MPS not available")

    return device_str


@pytest.fixture
def device(test_device):
    """Provide device string for individual tests."""
    return test_device
