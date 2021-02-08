# Copyright (c) 2020 ETH Zurich
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

source $SPACK_ROOT/share/spack/setup-env.sh

export CRAYPE_LINK_TYPE=dynamic
export APPS_ROOT="/apps/daint/SSL/HPX/packages"
export CLANG_VER="11.0.0"
export CXX_STD="20"
export BOOST_VER="1.75.0"
export HWLOC_VER="2.2.0"
export CLANG_ROOT="${APPS_ROOT}/llvm-${CLANG_VER}"
export BOOST_ROOT="${APPS_ROOT}/boost-${BOOST_VER}-clang-${CLANG_VER}-c++17-release"
export HWLOC_ROOT="${APPS_ROOT}/hwloc-${HWLOC_VER}-gcc-10.2.0"
export CXXFLAGS="-Wno-unused-command-line-argument -stdlib=libc++ -nostdinc++ -I${CLANG_ROOT}/include/c++/v1 -L${CLANG_ROOT}/lib -Wl,-rpath,${CLANG_ROOT}/lib,-lsupc++"
export LDCXXFLAGS="-stdlib=libc++ -L${CLANG_ROOT}/lib -Wl,-rpath,${CLANG_ROOT}/lib,-lsupc++"
export CXX="${CLANG_ROOT}/bin/clang++"
export CC="${CLANG_ROOT}/bin/clang"
export CPP="${CLANG_ROOT}/bin/clang -E"

module load daint-mc
spack load cmake
spack load ninja

configure_extra_options="-DCMAKE_BUILD_TYPE=Release"
configure_extra_options+=" -DHPX_WITH_MAX_CPU_COUNT=128"
configure_extra_options+=" -DHPX_WITH_MALLOC=system"
configure_extra_options+=" -DHPX_WITH_CXX${CXX_STD}=ON"
configure_extra_options+=" -DHPX_WITH_COMPILER_WARNINGS=ON"
configure_extra_options+=" -DHPX_WITH_COMPILER_WARNINGS_AS_ERRORS=ON"
configure_extra_options+=" -DHPX_WITH_SPINLOCK_DEADLOCK_DETECTION=ON"
configure_extra_options+=" -DHPX_WITH_UNITY_BUILD=ON"

# enable extra counters to verify everything compiles
configure_extra_options+=" -DHPX_WITH_BACKGROUND_THREAD_COUNTERS=ON"
configure_extra_options+=" -DHPX_WITH_COROUTINE_COUNTERS=ON"
configure_extra_options+=" -DHPX_WITH_PARCELPORT_ACTION_COUNTERS=ON"
configure_extra_options+=" -DHPX_WITH_THREAD_IDLE_RATES=ON"
configure_extra_options+=" -DHPX_WITH_THREAD_CREATION_AND_CLEANUP_RATES=ON"
configure_extra_options+=" -DHPX_WITH_THREAD_CUMULATIVE_COUNTS=ON"
configure_extra_options+=" -DHPX_WITH_THREAD_QUEUE_WAITTIME=ON"
configure_extra_options+=" -DHPX_WITH_THREAD_STEALING_COUNTS=ON"
