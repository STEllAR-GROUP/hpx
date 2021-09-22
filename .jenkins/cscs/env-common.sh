# Copyright (c) 2021 ETH Zurich
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

source $SPACK_ROOT/share/spack/setup-env.sh

spack load ccache@3.7.9
spack load cmake@3.18.6
spack load ninja@1.10.0

export CMAKE_CXX_COMPILER_LAUNCHER=ccache
export CMAKE_GENERATOR=Ninja
export CCACHE_DIR=/scratch/snx3000/simbergm/ccache-jenkins-hpx
export CCACHE_MAXSIZE=100G
export CCACHE_MAXFILES=50000

configure_extra_options+=" -DCMAKE_BUILD_TYPE=${build_type}"
configure_extra_options+=" -DHPX_WITH_CHECK_MODULE_DEPENDENCIES=ON"
