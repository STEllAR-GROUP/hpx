# Copyright (c) 2020 ETH Zurich
# Copyright (c) 2022 Hartmut Kaiser
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
set -eu

module purge
module load cmake
module load llvm/13
module load boost/1.78.0-Release
module load hwloc
module load openmpi

export CXX_STD="20"
