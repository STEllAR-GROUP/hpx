# Copyright (c) 2020 ETH Zurich
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

module purge
module load cmake
module load gcc/9
module load boost/1.73.0-${build_type,,}
module load hwloc
module load cuda/11.4
module load openmpi

export CXX_STD="17"

configure_extra_options+=" -DHPXLocal_CXX_STANDARD=${CXX_STD}"
configure_extra_options+=" -DHPXLocal_WITH_MALLOC=system"
configure_extra_options+=" -DHPXLocal_WITH_FETCH_ASIO=ON"
configure_extra_options+=" -DHPXLocal_WITH_COMPILER_WARNINGS=ON"
configure_extra_options+=" -DHPXLocal_WITH_COMPILER_WARNINGS_AS_ERRORS=OFF"
configure_extra_options+=" -DHPXLocal_WITH_CUDA=ON"
configure_extra_options+=" -DHPXLocal_WITH_DISTRIBUTED_RUNTIME=OFF"
configure_extra_options+=" -DHPXLocal_WITH_ASYNC_MPI=ON"
configure_extra_options+=" -DCMAKE_CUDA_ARCHITECTURES='37;70'"
