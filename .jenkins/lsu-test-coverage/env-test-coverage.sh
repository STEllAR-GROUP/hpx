# Copyright (c) 2023 Panos Syskakis
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
set -eu

module avail
module purge
module load cmake
module load gcc/12
module load boost/1.79.0-debug
module load hwloc
module load openmpi

export CCACHE_EXE=/work/pansysk75/ccache-4.12.2/bin/ccache
export CCACHE_DIR=/work/pansysk75/ccache-4.12.2/cache
export CCACHE_MAXSIZE=500G
export CCACHE_NOHASHDIR=1
