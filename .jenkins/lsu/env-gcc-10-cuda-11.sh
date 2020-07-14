# Copyright (c) 2020 ETH Zurich
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

module purge
module load cmake
module load gcc/9
module load boost/1.73.0-${build_type,,}
module load hwloc/2
module load cuda/11
module load mpich

configure_extra_options="-DCMAKE_BUILD_TYPE=${build_type}"
configure_extra_options+=" -DHPX_WITH_MALLOC=system"
configure_extra_options+=" -DHPX_WITH_DEPRECATION_WARNINGS=OFF"
configure_extra_options+=" -DHPX_WITH_COMPILER_WARNINGS=ON"
configure_extra_options+=" -DHPX_WITH_COMPILER_WARNINGS_AS_ERRORS=OFF"
configure_extra_options+=" -DHPX_WITH_CUDA=ON"
configure_extra_options+=" -DHPX_WITH_NETWORKING=OFF"
configure_extra_options+=" -DHPX_WITH_DISTRIBUTED_RUNTIME=OFF"
configure_extra_options+=" -DHPX_WITH_LOGGING=OFF"
configure_extra_options+=" -DHPX_WITH_ASYNC_MPI=ON"
