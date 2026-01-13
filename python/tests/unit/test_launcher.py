# HPXPy Launcher Tests
#
# SPDX-License-Identifier: BSL-1.0

"""Unit tests for HPXPy multi-locality launcher."""

import os
import pytest
import sys
from unittest.mock import patch, MagicMock


class TestLocalityConfig:
    """Test LocalityConfig class."""

    def test_locality_config_creation(self):
        """LocalityConfig should store configuration correctly."""
        from hpxpy.launcher import LocalityConfig

        config = LocalityConfig(
            locality_id=0,
            num_localities=4,
            host="localhost",
            port=7910,
            agas_host="localhost",
            agas_port=7910,
            num_threads=2,
        )

        assert config.locality_id == 0
        assert config.num_localities == 4
        assert config.host == "localhost"
        assert config.port == 7910
        assert config.num_threads == 2

    def test_locality_0_hpx_args(self):
        """Locality 0 should include AGAS server args."""
        from hpxpy.launcher import LocalityConfig

        config = LocalityConfig(
            locality_id=0,
            num_localities=2,
            host="localhost",
            port=7910,
            agas_host="localhost",
            agas_port=7910,
        )

        args = config.to_hpx_args()
        assert "--hpx:localities=2" in args
        assert "--hpx:hpx=localhost:7910" in args
        assert "--hpx:agas=localhost:7910" in args
        assert "--hpx:worker" not in args

    def test_worker_locality_hpx_args(self):
        """Worker localities should include --hpx:worker flag."""
        from hpxpy.launcher import LocalityConfig

        config = LocalityConfig(
            locality_id=1,
            num_localities=2,
            host="localhost",
            port=7911,
            agas_host="localhost",
            agas_port=7910,
        )

        args = config.to_hpx_args()
        assert "--hpx:localities=2" in args
        assert "--hpx:hpx=localhost:7911" in args
        assert "--hpx:agas=localhost:7910" in args
        assert "--hpx:worker" in args

    def test_threads_arg_included_when_nonzero(self):
        """Threads arg should be included when num_threads > 0."""
        from hpxpy.launcher import LocalityConfig

        config = LocalityConfig(
            locality_id=0,
            num_localities=1,
            num_threads=4,
        )

        args = config.to_hpx_args()
        assert "--hpx:threads=4" in args

    def test_threads_arg_excluded_when_zero(self):
        """Threads arg should be excluded when num_threads == 0."""
        from hpxpy.launcher import LocalityConfig

        config = LocalityConfig(
            locality_id=0,
            num_localities=1,
            num_threads=0,
        )

        args = config.to_hpx_args()
        assert not any("--hpx:threads" in arg for arg in args)


class TestLaunchConfig:
    """Test LaunchConfig class."""

    def test_launch_config_defaults(self):
        """LaunchConfig should have sensible defaults."""
        from hpxpy.launcher import LaunchConfig

        config = LaunchConfig()
        assert config.num_localities == 2
        assert config.base_port == 7910
        assert config.host == "localhost"

    def test_get_locality_config(self):
        """get_locality_config should return correct LocalityConfig."""
        from hpxpy.launcher import LaunchConfig

        config = LaunchConfig(
            num_localities=4,
            base_port=8000,
            host="127.0.0.1",
        )

        loc_config = config.get_locality_config(2)
        assert loc_config.locality_id == 2
        assert loc_config.num_localities == 4
        assert loc_config.port == 8002  # base_port + locality_id
        assert loc_config.agas_port == 8000  # Always base_port


class TestParseHpxArgs:
    """Test command-line argument parsing."""

    def test_parse_with_separator(self):
        """Should separate script args from HPX args."""
        from hpxpy.launcher import parse_hpx_args

        script_args, hpx_args = parse_hpx_args(
            ["--input", "data.csv", "--", "--hpx:localities=4"]
        )

        assert script_args == ["--input", "data.csv"]
        assert hpx_args == ["--hpx:localities=4"]

    def test_parse_without_separator(self):
        """Should return all args as script args if no separator."""
        from hpxpy.launcher import parse_hpx_args

        script_args, hpx_args = parse_hpx_args(["--input", "data.csv"])

        assert script_args == ["--input", "data.csv"]
        assert hpx_args == []

    def test_parse_empty_args(self):
        """Should handle empty args."""
        from hpxpy.launcher import parse_hpx_args

        script_args, hpx_args = parse_hpx_args([])

        assert script_args == []
        assert hpx_args == []


class TestEnvironmentFunctions:
    """Test environment-based helper functions."""

    def test_is_multi_locality_mode_false(self):
        """Should return False when not in multi-locality mode."""
        from hpxpy.launcher import is_multi_locality_mode

        with patch.dict(os.environ, {}, clear=True):
            assert is_multi_locality_mode() is False

    def test_is_multi_locality_mode_true(self):
        """Should return True when HPXPY_MULTI_LOCALITY is set."""
        from hpxpy.launcher import is_multi_locality_mode

        with patch.dict(os.environ, {"HPXPY_MULTI_LOCALITY": "1"}):
            assert is_multi_locality_mode() is True

    def test_get_expected_num_localities_default(self):
        """Should return 1 when not in multi-locality mode."""
        from hpxpy.launcher import get_expected_num_localities

        with patch.dict(os.environ, {}, clear=True):
            assert get_expected_num_localities() == 1

    def test_get_expected_num_localities_from_env(self):
        """Should return value from HPXPY_NUM_LOCALITIES."""
        from hpxpy.launcher import get_expected_num_localities

        with patch.dict(os.environ, {"HPXPY_NUM_LOCALITIES": "4"}):
            assert get_expected_num_localities() == 4


class TestFindFreePort:
    """Test port finding functionality."""

    def test_find_free_port_returns_int(self):
        """find_free_port should return an integer."""
        from hpxpy.launcher import find_free_port

        port = find_free_port(start_port=10000, num_ports=1)
        assert isinstance(port, int)
        assert port >= 10000

    def test_find_free_port_multiple_ports(self):
        """find_free_port should find consecutive ports."""
        from hpxpy.launcher import find_free_port

        port = find_free_port(start_port=10000, num_ports=4)
        assert isinstance(port, int)
        # The returned port should be valid for num_ports consecutive ports


class TestSpmdMain:
    """Test the spmd_main decorator."""

    def test_spmd_main_in_multilocality_mode(self):
        """spmd_main should call function directly in multi-locality mode."""
        from hpxpy.launcher import spmd_main

        called = []

        @spmd_main(num_localities=2)
        def test_func():
            called.append(True)

        with patch.dict(os.environ, {"HPXPY_MULTI_LOCALITY": "1"}):
            test_func()

        assert called == [True]

    def test_spmd_main_stores_config(self):
        """spmd_main should store configuration."""
        from hpxpy.launcher import spmd_main

        decorator = spmd_main(
            num_localities=4,
            base_port=9000,
            threads_per_locality=2,
            verbose=True,
        )

        assert decorator.num_localities == 4
        assert decorator.base_port == 9000
        assert decorator.threads_per_locality == 2
        assert decorator.verbose is True
