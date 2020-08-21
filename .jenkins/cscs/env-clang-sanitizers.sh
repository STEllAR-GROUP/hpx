# Copyright (c) 2020 ETH Zurich
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

source $SPACK_ROOT/share/spack/setup-env.sh

export CRAYPE_LINK_TYPE=dynamic
export APPS_ROOT="/apps/daint/SSL/HPX/packages"
export CLANG_VER="10.0.0"
export CXX_STD="17"
export BOOST_VER="1.73.0"
export HWLOC_VER="2.2.0"
export CLANG_ROOT="${APPS_ROOT}/llvm-${CLANG_VER}"
export BOOST_ROOT="${APPS_ROOT}/boost-${BOOST_VER}-clang-${CLANG_VER}-c++${CXX_STD}-debug"
export HWLOC_ROOT="${APPS_ROOT}/hwloc-${HWLOC_VER}-clang-${CLANG_VER}"
export CXXFLAGS="-fsanitize=address -fsanitize=undefined -fno-omit-frame-pointer -Wno-unused-command-line-argument -stdlib=libc++ -nostdinc++ -I${CLANG_ROOT}/include/c++/v1 -L${CLANG_ROOT}/lib -Wl,-rpath,${CLANG_ROOT}/lib,-lsupc++"
export LDCXXFLAGS="-fsanitize=address -stdlib=libc++ -L${CLANG_ROOT}/lib -Wl,-rpath,${CLANG_ROOT}/lib,-lsupc++"
export CXX="${CLANG_ROOT}/bin/clang++"
export CC="${CLANG_ROOT}/bin/clang"
export CPP="${CLANG_ROOT}/bin/clang -E"
export ASAN_OPTIONS=fast_unwind_on_malloc=0
export UBSAN_OPTIONS=print_stacktrace=1
export PATH=$CLANG_ROOT/bin:$PATH

module load daint-mc
spack load cmake
spack load ninja

configure_extra_options="-DCMAKE_BUILD_TYPE=Debug"
configure_extra_options+=" -DHPX_WITH_MAX_CPU_COUNT=128"
configure_extra_options+=" -DHPX_WITH_MALLOC=system"
configure_extra_options+=" -DHPX_WITH_CXX${CXX_STD}=ON"
configure_extra_options+=" -DHPX_WITH_DEPRECATION_WARNINGS=OFF"
configure_extra_options+=" -DHPX_WITH_COMPILER_WARNINGS=ON"
configure_extra_options+=" -DHPX_WITH_COMPILER_WARNINGS_AS_ERRORS=ON"
configure_extra_options+=" -DHPX_WITH_SANITIZERS=ON"
configure_extra_options+=" -DHPX_WITH_STACKOVERFLOW_DETECTION=OFF"

