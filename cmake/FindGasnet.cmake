# Copyright (c)      2017 Thomas Heller
# Copyright (c)      2023 Christopher Taylor
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

macro(find_gasnet)

set(GASNET_MPI_FOUND FALSE)
set(GASNET_UDP_FOUND FALSE)
set(GASNET_SMP_FOUND FALSE)

find_package(PkgConfig REQUIRED QUIET COMPONENTS)

if(HPX_WITH_PARCELPORT_GASNET_MPI)
   pkg_check_modules(GASNET REQUIRED IMPORTED_TARGET GLOBAL gasnet-mpi-par)
   set(GASNET_MPI_FOUND TRUE)
   hpx_setup_mpi()
endif()

if(HPX_WITH_PARCELPORT_GASNET_UDP)
   pkg_check_modules(GASNET REQUIRED IMPORTED_TARGET GLOBAL gasnet-udp-par)
   set(GASNET_UDP_FOUND TRUE)
endif()

if(HPX_WITH_PARCELPORT_GASNET_SMP)
   pkg_search_module(GASNET REQUIRED IMPORTED_TARGET GLOBAL gasnet-smp-par)
   set(GASNET_SMP_FOUND TRUE)
endif()

target_link_directories(hpx_core PUBLIC ${GASNET_LIBRARY_DIRS})

endmacro()
