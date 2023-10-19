# Copyright (c) 2020 ETH Zurich
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

module purge
module load cmake
module load gcc/13
module load boost/1.82.0-${build_type,,}
module load hwloc
module load openmpi
module load pwrapi/1.1.1

export HPXRUN_RUNWRAPPER=srun
export CXX_STD="20"

configure_extra_options+=" -DHPX_WITH_CXX_STANDARD=${CXX_STD}"
configure_extra_options+=" -DHPX_WITH_MALLOC=system"
configure_extra_options+=" -DHPX_WITH_FETCH_ASIO=ON"
configure_extra_options+=" -DHPX_WITH_COMPILER_WARNINGS=ON"
configure_extra_options+=" -DHPX_WITH_COMPILER_WARNINGS_AS_ERRORS=ON"
configure_extra_options+=" -DHPX_WITH_PARCELPORT_MPI=ON"
configure_extra_options+=" -DHPX_WITH_PARCELPORT_LCI=ON"
configure_extra_options+=" -DHPX_WITH_FETCH_LCI=ON"
configure_extra_options+=" -DHPX_WITH_PARCELPORT_GASNET=ON"
configure_extra_options+=" -DHPX_WITH_FETCH_GASNET=ON"
configure_extra_options+=" -DHPX_WITH_PARCELPORT_GASNET_CONDUIT=smp"
configure_extra_options+=" -DHPX_WITH_DATAPAR_BACKEND=EVE"
configure_extra_options+=" -DHPX_WITH_FETCH_EVE=ON"
configure_extra_options+=" -DHPX_WITH_EVE_TAG=main"

# The pwrapi library still needs to be set up properly on rostam
# configure_extra_options+=" -DHPX_WITH_POWER_COUNTER=ON"
