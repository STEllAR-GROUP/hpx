# HPXPy Runtime Tests
#
# SPDX-License-Identifier: BSL-1.0

"""Unit tests for HPX runtime management."""

import pytest


class TestRuntimeBasics:
    """Test basic runtime initialization and finalization."""

    def test_is_running_after_init(self, hpx_runtime):
        """Runtime should be running after initialization."""
        assert hpx_runtime.is_running()

    def test_num_threads(self, hpx_runtime):
        """Should report correct number of threads."""
        num = hpx_runtime.num_threads()
        assert num >= 1
        assert isinstance(num, int)

    def test_num_localities(self, hpx_runtime):
        """Should report at least 1 locality."""
        num = hpx_runtime.num_localities()
        assert num >= 1
        assert isinstance(num, int)

    def test_locality_id(self, hpx_runtime):
        """Locality ID should be valid."""
        lid = hpx_runtime.locality_id()
        assert lid >= 0
        assert lid < hpx_runtime.num_localities()


class TestRuntimeContextManager:
    """Test runtime context manager.

    Note: These tests run separately from the session fixture
    to test the context manager independently.
    """

    @pytest.mark.skip(reason="Cannot test context manager while session fixture is active")
    def test_context_manager_basic(self):
        """Test basic context manager usage."""
        import hpxpy as hpx

        with hpx.runtime(num_threads=2):
            assert hpx.is_running()
            assert hpx.num_threads() >= 1

        assert not hpx.is_running()

    @pytest.mark.skip(reason="Cannot test context manager while session fixture is active")
    def test_context_manager_exception(self):
        """Runtime should finalize even if exception occurs."""
        import hpxpy as hpx

        try:
            with hpx.runtime():
                raise ValueError("Test exception")
        except ValueError:
            pass

        assert not hpx.is_running()


class TestRuntimeErrors:
    """Test runtime error handling."""

    def test_double_init_raises(self, hpx_runtime):
        """Calling init twice should raise an error."""
        with pytest.raises(RuntimeError, match="already initialized"):
            hpx_runtime.init()

    @pytest.mark.skip(reason="Cannot test finalize while session fixture is active")
    def test_finalize_without_init_raises(self):
        """Calling finalize without init should raise an error."""
        import hpxpy as hpx

        with pytest.raises(RuntimeError, match="not initialized"):
            hpx.finalize()
