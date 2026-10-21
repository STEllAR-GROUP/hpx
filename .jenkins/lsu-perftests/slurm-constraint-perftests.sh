# Copyright (c) 2020 ETH Zurich
# Copyright (c) 2022 Hartmut Kaiser
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

configuration_slurm_partition="jenkins-compute"
# The nodes the performance tests may run on. entry.sh uses the first one that
# is idle, the ones with reference results first, or waits for the first one if
# none is. The build matrix and the coverage lane exclude the first one through
# configuration_slurm_exclude in their Slurm configuration files. Change those
# too if the performance tests move to another node. Each node needs its own
# reference results, see tools/perftests_ci/pack_references.sh.
configuration_slurm_nodes="medusa08 medusa00 medusa01 medusa02 medusa03
    medusa04 medusa05 medusa06 medusa07 medusa09 medusa10 medusa11 medusa12
    medusa13 medusa14 medusa15"
