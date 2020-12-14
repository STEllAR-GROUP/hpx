# Copyright (c) 2020 ETH Zurich
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

source $SPACK_ROOT/share/spack/setup-env.sh

export CRAYPE_LINK_TYPE=dynamic
export APPS_ROOT="/apps/daint/SSL/HPX/packages"
export GCC_VER="10.1.0"
export CXX_STD="17"
export BOOST_VER="1.75.0"
export HWLOC_VER="2.2.0"
export GCC_ROOT="${APPS_ROOT}/gcc-${GCC_VER}"
export BOOST_ROOT="${APPS_ROOT}/boost-${BOOST_VER}-gcc-${GCC_VER}-c++${CXX_STD}-debug"
export HWLOC_ROOT="${APPS_ROOT}/hwloc-${HWLOC_VER}-gcc-${GCC_VER}"
export CXXFLAGS="-nostdinc++ -I${GCC_ROOT}/include/c++/${GCC_VER} -I${GCC_ROOT}/include/c++/${GCC_VER}/x86_64-unknown-linux-gnu -I${GCC_ROOT}/include/c++/${GCC_VER}/x86_64-pc-linux-gnu -L${GCC_ROOT}/lib64 -Wl,-rpath,${GCC_ROOT}/lib64"
export LDFLAGS="-L${GCC_ROOT}/lib64"
export CXX=${GCC_ROOT}/bin/g++
export CC=${GCC_ROOT}/bin/gcc

module load daint-mc
spack load cmake
spack load ninja

configure_extra_options="-DCMAKE_BUILD_TYPE=Debug"
configure_extra_options+=" -DHPX_WITH_MAX_CPU_COUNT=128"
configure_extra_options+=" -DHPX_WITH_MALLOC=system"
configure_extra_options+=" -DHPX_WITH_CXX${CXX_STD}=ON"
configure_extra_options+=" -DHPX_WITH_NETWORKING=OFF"
configure_extra_options+=" -DHPX_WITH_DISTRIBUTED_RUNTIME=OFF"
configure_extra_options+=" -DHPX_WITH_LOGGING=OFF"
configure_extra_options+=" -DHPX_WITH_COMPILER_WARNINGS=ON"
configure_extra_options+=" -DHPX_WITH_COMPILER_WARNINGS_AS_ERRORS=OFF"
configure_extra_options+=" -DHPX_WITH_SPINLOCK_DEADLOCK_DETECTION=ON"
