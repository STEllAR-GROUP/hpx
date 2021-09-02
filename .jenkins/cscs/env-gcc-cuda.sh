# Copyright (c) 2020 ETH Zurich
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

export CRAYPE_LINK_TYPE=dynamic
export CXX_STD="17"

module load daint-gpu
module switch PrgEnv-cray PrgEnv-gnu
module load cudatoolkit
module load Boost/1.75.0-CrayGNU-20.11
module load hwloc/.2.0.3
spack load cmake@3.18.6
spack load ninja@1.10.0

export CXX=`which CC`
export CC=`which cc`

configure_extra_options+=" -DHPXLocal_CXX_STANDARD=${CXX_STD}"
configure_extra_options+=" -DHPXLocal_WITH_MALLOC=system"
configure_extra_options+=" -DHPXLocal_WITH_FETCH_ASIO=ON"
configure_extra_options+=" -DHPXLocal_WITH_CUDA=ON"
configure_extra_options+=" -DHPXLocal_WITH_EXAMPLES_OPENMP=ON"
configure_extra_options+=" -DHPXLocal_WITH_COMPILER_WARNINGS=ON"
configure_extra_options+=" -DHPXLocal_WITH_COMPILER_WARNINGS_AS_ERRORS=ON"
configure_extra_options+=" -DHPXLocal_WITH_TESTS_HEADERS=ON"
configure_extra_options+=" -DHWLOC_ROOT=${EBROOTHWLOC}"
