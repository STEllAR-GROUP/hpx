# Copyright (c) 2019-2023 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

if(HPX_WITH_NETWORKING AND HPX_WITH_PARCELPORT_GASNET)
  # setup GASNET, if necessary
  include(HPX_SetupGASNET)
  hpx_setup_gasnet()
else()
  return()
endif()

# Default location is $HPX_ROOT/libs/gasnet_base/include
set(gasnet_base_headers hpx/gasnet_base/gasnet.hpp hpx/gasnet_base/gasnet_environment.hpp)

# Default location is $HPX_ROOT/libs/gasnet_base/include_compatibility
# cmake-format: off
set(gasnet_base_compat_headers
    hpx/plugins/parcelport/gasnet/gasnet.hpp => hpx/modules/gasnet_base.hpp
    hpx/plugins/parcelport/gasnet/gasnet_environment.hpp => hpx/modules/gasnet_base.hpp
)
# cmake-format: on

set(gasnet_base_sources gasnet_environment.cpp)

include(HPX_AddModule)
add_hpx_module(
  core gasnet_base
  GLOBAL_HEADER_GEN ON
  SOURCES ${gasnet_base_sources}
  HEADERS ${gasnet_base_headers}
  COMPAT_HEADERS ${gasnet_base_compat_headers}
  MODULE_DEPENDENCIES gasnet hpx_logging hpx_runtime_configuration hpx_util
  DEPENDENCIES ${additional_dependencies} gasnet
  CMAKE_SUBDIRS examples tests
)
