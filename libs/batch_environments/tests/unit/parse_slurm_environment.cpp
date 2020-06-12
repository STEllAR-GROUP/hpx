//  Copyright (c) 2019 Marco Diers
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/batch_environments/slurm_environment.hpp>
#include <hpx/modules/testing.hpp>

#include <array>
#include <cstddef>
#include <string>
#include <utility>
#include <vector>

static constexpr bool enable_debug = false;

// example values in slurm conf
//   SelectTypeParameters=CR_Socket
//   NodeName=anode1 CPUs=48 Sockets=4 CoresPerSocket=6 ThreadsPerCore=2
//   NodeName=anode2 CPUs=16 Sockets=1 CoresPerSocket=8 ThreadsPerCore=2
//   NodeName=inode[7,14,20] CPUs=8 Sockets=1 CoresPerSocket=4 ThreadsPerCore=2
//   NodeName=inode11 CPUs=4 Sockets=1 CoresPerSocket=4 ThreadsPerCore=1

static auto run_in_slurm_env(
    std::vector<std::pair<const char*, const char*>>&& environment,
    std::vector<std::size_t>&& num_threads) -> void
{
    static const std::vector<std::string> hpx_nodelist{
        "anode1", "anode2", "inode7", "inode11", "inode14", "inode20"};
    static constexpr std::array<const char*, 13u> procids{
        {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12"}};
    static constexpr std::array<const char*, 13u> nodeids{
        {"0", "0", "0", "1", "1", "2", "2", "3", "3", "4", "4", "5", "5"}};

    for (auto& procid : procids)
    {
        auto index(static_cast<std::size_t>(std::atoi(procid)));
        std::vector<std::pair<const char*, const char*>> defaultenvironment{
            {"SLURM_STEP_NUM_TASKS", "13"},
            {"SLURM_STEP_NODELIST", "anode[1-2],inode[7,11,14,20]"},
            {"SLURM_STEP_TASKS_PER_NODE", "3,2(x5)"}, {"SLURM_PROCID", procid},
            {"SLURM_NODEID", nodeids[index]}};
        std::copy(std::begin(environment), std::end(environment),
            std::back_inserter(defaultenvironment));
        for (auto& env : defaultenvironment)
        {
            ::setenv(env.first, env.second, 1);
        }

        std::vector<std::string> nodelist;
        hpx::util::batch_environments::slurm_environment env(
            nodelist, enable_debug);
        HPX_TEST_EQ(env.valid(), true);
        HPX_TEST_EQ(nodelist == hpx_nodelist, true);
        HPX_TEST_EQ(env.node_num(), index);
        HPX_TEST_EQ(env.num_threads(), num_threads[index]);
        HPX_TEST_EQ(env.num_localities(), 13u);

        for (auto& env : defaultenvironment)
        {
            ::unsetenv(env.first);
        }
    }
    return;
}

// srun
//    use the defaults
static auto default_test() -> void
{
    run_in_slurm_env({{"SLURM_JOB_CPUS_PER_NODE", "12,16,8,4,8(x2)"}},
        {4u, 4u, 4u, 8u, 8u, 4u, 4u, 2u, 2u, 4u, 4u, 4u, 4u});
    return;
}

// srun --exclusive
//    affected environment variable SLURM_JOB_CPUS_PER_NODE
static auto exclusive_test() -> void
{
    run_in_slurm_env({{"SLURM_JOB_CPUS_PER_NODE", "48,16,8,4,8(x2)"}},
        {16u, 16u, 16, 8u, 8u, 4u, 4u, 2u, 2u, 4u, 4u, 4u, 4u});
    return;
}

// srun --cpus-per-task=1
//    set environment variable SLURM_CPUS_PER_TASK=1
static auto cpuspertask_test() -> void
{
    run_in_slurm_env({{"SLURM_CPUS_PER_TASK", "1"},
                         {"SLURM_JOB_CPUS_PER_NODE", "12,16,8,4,8(x2)"}},
        {1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u});
    return;
}

// srun --exclusive --cpus-per-task=1
//    set environment variable SLURM_CPUS_PER_TASK=1
//    affected environment variable SLURM_JOB_CPUS_PER_NODE
static auto cpuspertask_exclusive_test() -> void
{
    run_in_slurm_env({{"SLURM_CPUS_PER_TASK", "1"},
                         {"SLURM_JOB_CPUS_PER_NODE", "48,16,8,4,8(x2)"}},
        {1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u});
    return;
}

// srun --threads-per-core=1
//    affected environment variable SLURM_JOB_CPUS_PER_NODE
static auto threadspercore_test() -> void
{
    run_in_slurm_env({{"SLURM_JOB_CPUS_PER_NODE", "12,16,8,4,8(x2)"}},
        {4u, 4u, 4u, 8u, 8u, 4u, 4u, 2u, 2u, 4u, 4u, 4u, 4u});
    return;
}

// srun --exclusive --threads-per-core=1
//    affected environment variable SLURM_JOB_CPUS_PER_NODE
static auto threadspercore_exclusive_test() -> void
{
    run_in_slurm_env({{"SLURM_JOB_CPUS_PER_NODE", "24,8,4(x4)"}},
        {8u, 8u, 8u, 4u, 4u, 2u, 2u, 2u, 2u, 2u, 2u, 2u, 2u});
    return;
}

int main()
{
    // disabled for run in slurm environment
    if (std::getenv("SLURM_PROCID"))
    {
        return 0;
    }
    default_test();
    exclusive_test();
    cpuspertask_test();
    cpuspertask_exclusive_test();
    threadspercore_test();
    threadspercore_exclusive_test();
    return 0;
}
