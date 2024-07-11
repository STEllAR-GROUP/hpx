# Copyright (c) 2020 ETH Zurich
# Copyright (c) 2024 Alireza Kheirkhahan
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

module purge
module load cmake
module load gcc/11
module load boost/1.78.0-${build_type,,}
module load hwloc
module load openmpi

export HPXRUN_RUNWRAPPER=srun
export CXX_STD="20"

configure_extra_options+=" -DCMAKE_BUILD_TYPE=${build_type}"
configure_extra_options+=" -DHPX_WITH_CXX_STANDARD=${CXX_STD}"
configure_extra_options+=" -DHPX_WITH_MALLOC=system"
configure_extra_options+=" -DHPX_WITH_FETCH_ASIO=ON"
configure_extra_options+=" -DHPX_WITH_COMPILER_WARNINGS=ON"
configure_extra_options+=" -DHPX_WITH_COMPILER_WARNINGS_AS_ERRORS=ON"
configure_extra_options+=" -DHPX_WITH_NETWORKING=OFF"
configure_extra_options+=" -DHPX_WITH_DATAPAR_BACKEND=STD_EXPERIMENTAL_SIMD"

# The pwrapi library still needs to be set up properly on rostam
# configure_extra_options+=" -DHPX_WITH_POWER_COUNTER=ON"

configure_extra_options+=" -DHPX_WITH_TESTS_COMMAND_LINE=--hpx:queuing=local-workrequesting-fifo"
