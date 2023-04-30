# Copyright (c) 2019-2023 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

if(NOT
   (HPX_WITH_NETWORKING
    AND HPX_WITH_PARCELPORT_GASNET
    AND HPX_WITH_PARCELPORT_GASNET_CONDUIT)
)
  return()
endif()

include(HPX_SetupGasnet)
hpx_setup_gasnet()

# Default location is $HPX_ROOT/libs/gasnet_base/include
set(gasnet_base_headers hpx/gasnet_base/gasnet.hpp
                        hpx/gasnet_base/gasnet_environment.hpp
)

set(gasnet_base_sources gasnet_environment.cpp)

if("${HPX_WITH_PARCELPORT_GASNET_CONDUIT}" STREQUAL "mpi")
  set(gasnet_additional_dependencies Mpi::mpi)
endif()

include(HPX_AddModule)
add_hpx_module(
  core gasnet_base
  GLOBAL_HEADER_GEN ON
  SOURCES ${gasnet_base_sources}
  HEADERS ${gasnet_base_headers}
  DEPENDENCIES PkgConfig::GASNET ${gasnet_additional_dependencies}
  MODULE_DEPENDENCIES hpx_logging hpx_runtime_configuration hpx_string_util
                      hpx_threading_base hpx_util
  CMAKE_SUBDIRS examples tests
)
