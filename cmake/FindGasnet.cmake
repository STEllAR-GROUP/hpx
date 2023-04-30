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

if(HPX_WITH_NETWORKING)
   pkg_check_modules(GASNET QUIET gasnet-mpi-seq)
   if(NOT GASNET_FOUND)
      pkg_check_modules(PC_GASNET QUIET gasnet-udp-seq)
      if(NOT GASNET_FOUND)
         hpx_error("Could not find GASNET MPI/udp please set the PKG_CONFIG_PATH environment variable")
      else()
         set(GASNET_UDP_FOUND TRUE)
      endif()
   else()
      set(GASNET_MPI_FOUND TRUE)
      hpx_setup_mpi()
   endif()
else()
   pkg_check_modules(GASNET QUIET gasnet-smp-seq)
   if(NOT GASNET_FOUND)
      hpx_error("Could not find GASNET please set the PKG_CONFIG_PATH environment variable")
   else()
      set(GASNET_SMP_FOUND TRUE)
   endif()
endif()

if(NOT GASNET_INCLUDE_DIRS OR NOT GASNET_LIBRARY_DIRS)
  hpx_error("Could not find GASNET_INCLUDE_DIRS or GASNET_LIBRARY_DIRS please \
  set the PKG_CONFIG_PATH environment variable"
  )
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
  Gasnet DEFAULT_MSG GASNET_LIBRARY_DIRS GASNET_INCLUDE_DIRS
)

endmacro()
