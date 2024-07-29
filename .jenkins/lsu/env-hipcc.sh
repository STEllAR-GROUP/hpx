# Copyright (c) 2020-2021 ETH Zurich
# Copyright (c) 2024 Alireza Kheirkhahan
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)


module purge
module load cmake
module load rocm
module load boost/1.84.0-${build_type,,}
#export CXX=hipcc

export HPXRUN_RUNWRAPPER=srun
export CXX_STD="20"

configure_extra_options+=" -DHPX_WITH_CXX_STANDARD=${CXX_STD}"
configure_extra_options+=" -DHPX_WITH_MALLOC=system"
configure_extra_options+=" -DHPX_WITH_FETCH_ASIO=ON"
configure_extra_options+=" -DHPX_WITH_MAX_CPU_COUNT=128"
configure_extra_options+=" -DHPX_WITH_DEPRECATION_WARNINGS=OFF"
configure_extra_options+=" -DHPX_WITH_COMPILER_WARNINGS=ON"
configure_extra_options+=" -DHPX_WITH_COMPILER_WARNINGS_AS_ERRORS=OFF"

ctest_extra_args+=" -E tests.unit.modules.algorithms.detail "
ctest_extra_args+=" -E tests.regressions.modules.coroutines.coroutine_function_destructor_yield_4800"
