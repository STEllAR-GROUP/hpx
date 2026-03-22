# HPX Task-Bench Example

This directory contains an HPX implementation of the [Task-Bench](https://github.com/StanfordLegion/task-bench) benchmark suite.

Task-Bench is a configurable benchmark for evaluating the efficiency and performance of task-based runtime systems. For more details on the benchmark logic and configuration, please refer to the original [Task-Bench Paper](https://slac.stanford.edu/~slat/papers/taskbench_sc19.pdf).

## Features

- **Portability**: Includes the full Task-Bench core logic (`core.cc`, `core.h`, etc.) for standalone execution.
- **HPX Optimization**: Utilizes `hpx::experimental::for_loop` with a `fork_join_executor` to efficiently handle dependencies and point executions.
- **Configurable Patterns**: Supports all standard Task-Bench dependence patterns:
  - `trivial`, `no_comm`, `stencil_1d`, `stencil_1d_periodic`, `dom`, `tree`, `fft`, `all_to_all`, `nearest`, `spread`, `random_nearest`.
- **Flexible Kernels**: Supports various execution kernels:
  - `empty`, `busy_wait`, `compute_bound`, `compute_bound2`, `memory_bound`.
- **Distributed Ready**: Base infrastructure for MPI-based distributed execution (uses a mock layer for single-node builds).

## Building

This example is integrated into the HPX build system. To build it, ensure `HPX_WITH_EXAMPLES=ON` is set in your CMake configuration.

```bash
# From your build directory
make task_bench
```

## Usage

Run the benchmark using the following command format:

```bash
./bin/task_bench -steps <num_steps> -width <num_tasks> -type <dependence_type> [options]
```

### Examples

**Running a 1D Stencil with 10 steps and 8 tasks:**
```bash
./bin/task_bench -steps 10 -width 8 -type stencil_1d
```

**Running an FFT pattern with a busy-wait kernel:**
```bash
./bin/task_bench -steps 10 -width 16 -type fft -kernel busy_wait -iter 1000
```

**Running a memory-bound kernel (requires scratch size):**
```bash
./bin/task_bench -steps 10 -width 4 -type dom -kernel memory_bound -scratch 1048576
```

## Options

- `-steps [INT]`: Number of timesteps.
- `-width [INT]`: Number of tasks per timestep.
- `-type [DEP]`: Dependence pattern (stencil_1d, fft, etc.).
- `-kernel [KERNEL]`: Execution kernel (busy_wait, compute_bound, etc.).
- `-iter [INT]`: Number of iterations for the kernel.
- `-scratch [INT]`: Scratch space size in bytes for memory kernels.
- `-h`: Show full help message.

## Implementation Notes

The HPX driver (`hpx_task_bench.cpp`) implements a custom execution loop that minimizes overhead for shared-memory patterns while maintaining compatibility with the Task-Bench core logic. It identifies patterns that allow for optimized local communication and uses HPX's advanced parallel algorithms to maximize hardware utilization.
