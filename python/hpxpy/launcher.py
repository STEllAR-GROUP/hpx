# HPXPy - Multi-Locality Launcher
#
# SPDX-License-Identifier: BSL-1.0

"""
Multi-locality launcher for HPXPy.

This module provides utilities for launching HPX applications across
multiple processes (localities) with proper TCP parcelport configuration.

Basic usage::

    from hpxpy.launcher import launch_localities

    # Launch 4 localities running the same script
    launch_localities(
        script="my_distributed_app.py",
        num_localities=4,
    )

Or use the SPMD helper::

    from hpxpy.launcher import spmd_main

    @spmd_main(num_localities=4)
    def main():
        import hpxpy as hpx
        print(f"Locality {hpx.locality_id()} of {hpx.num_localities()}")
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence


@dataclass
class LocalityConfig:
    """Configuration for a single HPX locality.

    Attributes
    ----------
    locality_id : int
        The ID of this locality (0 is the root/AGAS server).
    num_localities : int
        Total number of localities in the distributed run.
    host : str
        Hostname/IP address for this locality.
    port : int
        Port number for HPX communication.
    agas_host : str
        Hostname/IP address of the AGAS server (locality 0).
    agas_port : int
        Port number of the AGAS server.
    num_threads : int
        Number of OS threads for this locality.
    """
    locality_id: int
    num_localities: int
    host: str = "localhost"
    port: int = 7910
    agas_host: str = "localhost"
    agas_port: int = 7910
    num_threads: int = 0  # 0 means use all available cores

    def to_hpx_args(self) -> list[str]:
        """Convert to HPX command-line arguments.

        Returns
        -------
        list of str
            HPX command-line arguments for this locality.
        """
        args = [
            f"--hpx:localities={self.num_localities}",
            f"--hpx:hpx={self.host}:{self.port}",
        ]

        if self.locality_id == 0:
            # Locality 0 runs the AGAS server
            args.append(f"--hpx:agas={self.agas_host}:{self.agas_port}")
        else:
            # Workers connect to locality 0's AGAS server
            args.append("--hpx:worker")
            args.append(f"--hpx:agas={self.agas_host}:{self.agas_port}")

        if self.num_threads > 0:
            args.append(f"--hpx:threads={self.num_threads}")

        return args


@dataclass
class LaunchConfig:
    """Configuration for launching multiple HPX localities.

    Attributes
    ----------
    num_localities : int
        Number of localities to launch.
    base_port : int
        Starting port number (each locality uses base_port + locality_id).
    host : str
        Hostname/IP address for all localities.
    threads_per_locality : int
        Number of threads per locality (0 for auto).
    env : dict
        Additional environment variables.
    """
    num_localities: int = 2
    base_port: int = 7910
    host: str = "localhost"
    threads_per_locality: int = 0
    env: dict = field(default_factory=dict)

    def get_locality_config(self, locality_id: int) -> LocalityConfig:
        """Get configuration for a specific locality.

        Parameters
        ----------
        locality_id : int
            The locality ID to configure.

        Returns
        -------
        LocalityConfig
            Configuration for the specified locality.
        """
        return LocalityConfig(
            locality_id=locality_id,
            num_localities=self.num_localities,
            host=self.host,
            port=self.base_port + locality_id,
            agas_host=self.host,
            agas_port=self.base_port,  # AGAS is always on locality 0's port
            num_threads=self.threads_per_locality,
        )


def find_free_port(start_port: int = 7910, num_ports: int = 1) -> int:
    """Find a range of free ports starting from a given port.

    Parameters
    ----------
    start_port : int
        Port to start searching from.
    num_ports : int
        Number of consecutive ports needed.

    Returns
    -------
    int
        Starting port of the free range.

    Raises
    ------
    RuntimeError
        If no suitable port range is found.
    """
    max_attempts = 100
    for attempt in range(max_attempts):
        port = start_port + attempt
        all_free = True
        for i in range(num_ports):
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(1)
                sock.bind(("localhost", port + i))
                sock.close()
            except OSError:
                all_free = False
                break
        if all_free:
            return port
    raise RuntimeError(f"Could not find {num_ports} consecutive free ports")


def launch_localities(
    script: str,
    num_localities: int = 2,
    base_port: int = 0,
    host: str = "localhost",
    threads_per_locality: int = 0,
    script_args: Sequence[str] = (),
    env: Optional[dict] = None,
    wait: bool = True,
    verbose: bool = False,
) -> list[subprocess.Popen]:
    """Launch a Python script across multiple HPX localities.

    This function spawns multiple Python processes, each configured as
    an HPX locality that can communicate via TCP parcelport.

    Parameters
    ----------
    script : str
        Path to the Python script to run.
    num_localities : int
        Number of localities to launch.
    base_port : int
        Starting port number. If 0, automatically find free ports.
    host : str
        Hostname/IP address for all localities.
    threads_per_locality : int
        Number of threads per locality (0 for auto).
    script_args : sequence of str
        Additional arguments to pass to the script.
    env : dict, optional
        Additional environment variables.
    wait : bool
        If True, wait for all processes to complete.
    verbose : bool
        If True, print launch information.

    Returns
    -------
    list of subprocess.Popen
        List of launched processes (empty if wait=True and completed).

    Examples
    --------
    >>> launch_localities("distributed_app.py", num_localities=4)
    >>> # Or with custom configuration
    >>> launch_localities(
    ...     "distributed_app.py",
    ...     num_localities=4,
    ...     threads_per_locality=2,
    ...     script_args=["--input", "data.csv"],
    ... )
    """
    if base_port == 0:
        base_port = find_free_port(7910, num_localities)

    config = LaunchConfig(
        num_localities=num_localities,
        base_port=base_port,
        host=host,
        threads_per_locality=threads_per_locality,
        env=env or {},
    )

    processes: list[subprocess.Popen] = []
    python_exe = sys.executable

    # Build environment
    launch_env = os.environ.copy()
    launch_env.update(config.env)
    # Set HPXPY_MULTI_LOCALITY to signal the script it's in distributed mode
    launch_env["HPXPY_MULTI_LOCALITY"] = "1"
    launch_env["HPXPY_NUM_LOCALITIES"] = str(num_localities)

    # Launch locality 0 first (AGAS server)
    # Then launch workers (localities 1 to N-1)
    for locality_id in range(num_localities):
        loc_config = config.get_locality_config(locality_id)
        hpx_args = loc_config.to_hpx_args()

        # Build command
        cmd = [python_exe, script]
        cmd.extend(script_args)
        cmd.append("--")  # Separator for HPX args
        cmd.extend(hpx_args)

        # Set locality-specific env vars
        proc_env = launch_env.copy()
        proc_env["HPXPY_LOCALITY_ID"] = str(locality_id)

        if verbose:
            print(f"[Launcher] Starting locality {locality_id}: {' '.join(cmd)}")

        proc = subprocess.Popen(
            cmd,
            env=proc_env,
            stdout=subprocess.PIPE if not verbose else None,
            stderr=subprocess.PIPE if not verbose else None,
        )
        processes.append(proc)

        # Give locality 0 a moment to start the AGAS server
        if locality_id == 0 and num_localities > 1:
            time.sleep(0.5)

    if wait:
        # Wait for all processes
        exit_codes = []
        for i, proc in enumerate(processes):
            code = proc.wait()
            exit_codes.append(code)
            if verbose:
                print(f"[Launcher] Locality {i} exited with code {code}")

        # Check for failures
        failures = [(i, c) for i, c in enumerate(exit_codes) if c != 0]
        if failures:
            raise RuntimeError(
                f"Some localities failed: {failures}"
            )
        return []

    return processes


def get_hpx_args_from_env() -> list[str]:
    """Get HPX arguments based on environment variables set by launcher.

    This function reads HPXPY_* environment variables set by the launcher
    and returns appropriate HPX configuration arguments.

    Returns
    -------
    list of str
        HPX command-line arguments based on environment, or empty list
        if not launched via the multi-locality launcher.
    """
    if os.environ.get("HPXPY_MULTI_LOCALITY") != "1":
        return []

    # These would be set by the launcher
    # The actual HPX args are passed via command line, so this is mainly
    # for informational purposes
    return []


def is_multi_locality_mode() -> bool:
    """Check if running in multi-locality mode.

    Returns
    -------
    bool
        True if launched via the multi-locality launcher.
    """
    return os.environ.get("HPXPY_MULTI_LOCALITY") == "1"


def get_expected_num_localities() -> int:
    """Get the expected number of localities from environment.

    Returns
    -------
    int
        Expected number of localities, or 1 if not in multi-locality mode.
    """
    return int(os.environ.get("HPXPY_NUM_LOCALITIES", "1"))


class spmd_main:
    """Decorator for SPMD (Single Program Multiple Data) main functions.

    This decorator handles launching multiple localities when the script
    is run directly, and just calls the function when run as a locality.

    Parameters
    ----------
    num_localities : int
        Number of localities to launch.
    base_port : int
        Starting port number (0 for auto).
    threads_per_locality : int
        Threads per locality (0 for auto).
    verbose : bool
        Print launch information.

    Examples
    --------
    >>> @spmd_main(num_localities=4)
    ... def main():
    ...     import hpxpy as hpx
    ...     with hpx.runtime():
    ...         print(f"Locality {hpx.locality_id()}")
    ...
    >>> if __name__ == "__main__":
    ...     main()
    """

    def __init__(
        self,
        num_localities: int = 2,
        base_port: int = 0,
        threads_per_locality: int = 0,
        verbose: bool = False,
    ):
        self.num_localities = num_localities
        self.base_port = base_port
        self.threads_per_locality = threads_per_locality
        self.verbose = verbose

    def __call__(self, func: Callable[[], None]) -> Callable[[], None]:
        """Wrap the main function."""
        def wrapper():
            if is_multi_locality_mode():
                # We're a spawned locality - just run the function
                func()
            else:
                # We're the launcher - spawn localities
                import __main__
                script = getattr(__main__, "__file__", None)
                if script is None:
                    raise RuntimeError(
                        "spmd_main can only be used in a script, not interactive mode"
                    )

                launch_localities(
                    script=script,
                    num_localities=self.num_localities,
                    base_port=self.base_port,
                    threads_per_locality=self.threads_per_locality,
                    verbose=self.verbose,
                )

        return wrapper


def parse_hpx_args(args: Optional[Sequence[str]] = None) -> tuple[list[str], list[str]]:
    """Parse command-line arguments to separate script args from HPX args.

    HPX arguments come after a '--' separator.

    Parameters
    ----------
    args : sequence of str, optional
        Command-line arguments. Defaults to sys.argv[1:].

    Returns
    -------
    tuple of (list of str, list of str)
        (script_args, hpx_args)

    Examples
    --------
    >>> parse_hpx_args(["--input", "data.csv", "--", "--hpx:localities=4"])
    (['--input', 'data.csv'], ['--hpx:localities=4'])
    """
    if args is None:
        args = sys.argv[1:]

    args = list(args)
    if "--" in args:
        sep_idx = args.index("--")
        return args[:sep_idx], args[sep_idx + 1:]
    return args, []


def init_from_args(args: Optional[Sequence[str]] = None) -> None:
    """Initialize HPX runtime from command-line arguments.

    This is a convenience function that parses HPX arguments from the
    command line and initializes the runtime.

    Parameters
    ----------
    args : sequence of str, optional
        Command-line arguments. Defaults to sys.argv[1:].

    Examples
    --------
    >>> # In a distributed script:
    >>> from hpxpy.launcher import init_from_args
    >>> init_from_args()  # Parses sys.argv for HPX args
    >>> # ... do distributed work ...
    >>> hpx.finalize()
    """
    import hpxpy as hpx

    _, hpx_args = parse_hpx_args(args)
    hpx.init(config=hpx_args)
