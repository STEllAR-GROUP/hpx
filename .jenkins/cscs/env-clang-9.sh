# Copyright (c) 2021 ETH Zurich
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

source $SPACK_ROOT/share/spack/setup-env.sh

export CRAYPE_LINK_TYPE=dynamic
export APPS_ROOT="/apps/daint/SSL/HPX/packages"
export CLANG_VER="9.0.1"
export CXX_STD="14"
export BOOST_VER="1.70.0"
export HWLOC_VER="1.11.11"
export CLANG_ROOT="${APPS_ROOT}/llvm-${CLANG_VER}"
export BOOST_ROOT="${APPS_ROOT}/boost-${BOOST_VER}-clang-${CLANG_VER}-c++${CXX_STD}-debug"
export HWLOC_ROOT="${APPS_ROOT}/hwloc-${HWLOC_VER}-gcc-10.2.0"
export CXXFLAGS="-Wno-unused-command-line-argument -stdlib=libc++ -nostdinc++ -I${CLANG_ROOT}/include/c++/v1 -L${CLANG_ROOT}/lib -Wl,-rpath,${CLANG_ROOT}/lib,-lsupc++"
export LDCXXFLAGS="-stdlib=libc++ -L${CLANG_ROOT}/lib -Wl,-rpath,${CLANG_ROOT}/lib,-lsupc++"
export CXX="${CLANG_ROOT}/bin/clang++"
export CC="${CLANG_ROOT}/bin/clang"
export CPP="${CLANG_ROOT}/bin/clang -E"

module load daint-mc
spack load cmake
spack load ninja

configure_extra_options="-DCMAKE_BUILD_TYPE=Debug"
configure_extra_options+=" -DHPX_WITH_MAX_CPU_COUNT=128"
configure_extra_options+=" -DHPX_WITH_MALLOC=system"
configure_extra_options+=" -DHPX_WITH_COMPILER_WARNINGS=ON"
configure_extra_options+=" -DHPX_WITH_COMPILER_WARNINGS_AS_ERRORS=ON"
