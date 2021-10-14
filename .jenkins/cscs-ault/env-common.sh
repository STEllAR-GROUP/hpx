# Copyright (c) 2021 ETH Zurich
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

source $SPACK_ROOT/share/spack/setup-env.sh

spack load ccache
# In the latest version of cmake module on ault (3.21.2), the
# compiler features set in cmake are discarded by the ROCMClang
# compiler (fixed in later versions)
# https://gitlab.kitware.com/cmake/cmake/-/issues/22460
spack load cmake@3.21.3
spack load ninja@1.10.2
# The openssl package loaded with cmake causes problems with ROCM
spack unload openssl

export CMAKE_CXX_COMPILER_LAUNCHER=ccache
export CMAKE_GENERATOR=Ninja
export CCACHE_DIR=/apps/ault/SSL/HPX/ccache/ccache-jenkins-hpx
export CCACHE_MAXSIZE=100G
export CCACHE_MAXFILES=50000

configure_extra_options+=" -DCMAKE_BUILD_TYPE=${build_type}"
configure_extra_options+=" -DHPX_WITH_CHECK_MODULE_DEPENDENCIES=ON"
