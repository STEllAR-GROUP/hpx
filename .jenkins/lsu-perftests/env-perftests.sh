# Copyright (c) 2020 ETH Zurich
# Copyright (c) 2022 Hartmut Kaiser
# Copyright (c) 2024 Alireza Kheirkhahan

#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
set -eu

module purge
module load cmake
module load llvm/18
module load boost/1.85.0-release
module load hwloc
module load openmpi

export CXX_STD="20"

export CCACHE_EXE=/work/pansysk75/ccache-4.12.2/bin/ccache
export CCACHE_DIR=/work/pansysk75/ccache-4.12.2/cache
export CCACHE_MAXSIZE=500G
export CCACHE_NOHASHDIR=1

export CMAKE_CXX_COMPILER_LAUNCHER=${CCACHE_EXE}
export CMAKE_C_COMPILER_LAUNCHER=${CCACHE_EXE}

