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
