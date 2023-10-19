# Copyright (c) 2023 Christopher Taylor
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

if(NOT
   (HPX_WITH_NETWORKING
    AND HPX_WITH_PARCELPORT_OPENSHMEM)
)
  return()
endif()

include(HPX_SetupOpenSHMEM)
hpx_setup_openshmem()

# Default location is $HPX_ROOT/libs/openshmem_base/include
set(openshmem_base_headers hpx/openshmem_base/openshmem.hpp
                           hpx/openshmem_base/openshmem_environment.hpp
)

set(openshmem_base_sources openshmem_environment.cpp)

include(HPX_AddModule)
add_hpx_module(
  core openshmem_base
  GLOBAL_HEADER_GEN ON
  SOURCES ${openshmem_base_sources}
  HEADERS ${openshmem_base_headers}
  DEPENDENCIES PkgConfig::OPENSHMEM
  MODULE_DEPENDENCIES hpx_logging hpx_runtime_configuration hpx_string_util
                      hpx_threading_base hpx_util
  CMAKE_SUBDIRS examples tests
)
