# Copyright (c) 2020 ETH Zurich
# Copyright (c) 2023 STE||AR-GROUP
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

module purge
module load cmake
module load llvm/12
module load boost/1.75.0-${build_type,,}
module load hwloc
module load openmpi
module load pwrapi/1.1.1
module load papi/7.0.1

export HPXRUN_RUNWRAPPER=srun
export CXX_STD="20"
export PKG_CONFIG_PATH=/usr/local/lib/pkgconfig:$PKG_CONFIG_PATH

configure_extra_options+=" -DHPX_WITH_MAX_CPU_COUNT=128"
configure_extra_options+=" -DHPX_WITH_MALLOC=system"
configure_extra_options+=" -DHPX_WITH_FETCH_ASIO=ON"
configure_extra_options+=" -DHPX_WITH_CXX_STANDARD=${CXX_STD}"
configure_extra_options+=" -DHPX_WITH_COMPILER_WARNINGS=ON"
configure_extra_options+=" -DHPX_WITH_COMPILER_WARNINGS_AS_ERRORS=ON"
configure_extra_options+=" -DHPX_WITH_SPINLOCK_DEADLOCK_DETECTION=ON"
configure_extra_options+=" -DHPX_WITH_APEX=ON"
configure_extra_options+=" -DHPX_WITH_FETCH_APEX=ON"
configure_extra_options+=" -DHPX_WITH_PAPI=ON"
configure_extra_options+=" -DAPEX_WITH_PAPI=ON"
#configure_extra_options+=" -DAPEX_WITH_OTF2=ON"
#configure_extra_options+=" -DOTF2_ROOT=${OTF2_ROOT}"
