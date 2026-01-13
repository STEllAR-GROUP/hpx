# HPXPy test configuration
#
# SPDX-License-Identifier: BSL-1.0

"""Pytest configuration and fixtures for HPXPy tests."""

import pytest


@pytest.fixture(scope="session")
def hpx_runtime():
    """Initialize HPX runtime for the entire test session.

    This fixture initializes the HPX runtime once at the beginning
    of the test session and finalizes it at the end.
    """
    import hpxpy as hpx

    hpx.init(num_threads=4)
    yield hpx
    hpx.finalize()


@pytest.fixture
def sample_array(hpx_runtime):
    """Create a sample array for testing."""
    import numpy as np

    hpx = hpx_runtime
    np_arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    return hpx.from_numpy(np_arr)


@pytest.fixture
def large_array(hpx_runtime):
    """Create a larger array for performance-sensitive tests."""
    import numpy as np

    hpx = hpx_runtime
    np_arr = np.random.randn(100000)
    return hpx.from_numpy(np_arr)
