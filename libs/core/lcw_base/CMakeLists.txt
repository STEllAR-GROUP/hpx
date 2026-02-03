# Copyright (c) 2019-2021 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

if(NOT (HPX_WITH_NETWORKING AND HPX_WITH_PARCELPORT_LCW))
  return()
endif()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# setup LCW, if necessary
include(HPX_SetupLCW)
hpx_setup_lcw()

# Default location is $HPX_ROOT/libs/lcw_base/include
set(lcw_base_headers hpx/lcw_base/lcw_environment.hpp)

set(lcw_base_sources lcw_environment.cpp)

include(HPX_AddModule)
add_hpx_module(
  core lcw_base
  GLOBAL_HEADER_GEN ON
  SOURCES ${lcw_base_sources}
  HEADERS ${lcw_base_headers}
  MODULE_DEPENDENCIES hpx_logging hpx_runtime_configuration hpx_string_util
                      hpx_util
  DEPENDENCIES LCW::LCW LCI::LCT
  CMAKE_SUBDIRS examples tests
)
