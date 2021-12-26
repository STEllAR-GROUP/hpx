# Copyright (c) 2019-2021 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(parcelset_headers hpx/parcelset/parcel.hpp hpx/parcelset/parcelhandler.hpp
                      hpx/parcelset/parcelset_fwd.hpp
)

# cmake-format: off
set(parcelset_compat_headers
    hpx/runtime/parcelset/parcel.hpp => hpx/parcelset/parcel.hpp
    hpx/runtime/parcelset/parcelhandler.hpp => hpx/parcelset/parcelhandler.hpp
    hpx/runtime/parcelset_fwd.hpp => hpx/parcelset/parcelset_fwd.hpp
)
# cmake-format: on

set(parcelset_sources parcel.cpp parcelhandler.cpp)

include(HPX_AddModule)
add_hpx_module(
  full parcelset
  GLOBAL_HEADER_GEN ON
  SOURCES ${parcelset_sources}
  HEADERS ${parcelset_headers}
  COMPAT_HEADERS ${parcelset_compat_headers}
  DEPENDENCIES HPX::hpx_local
  MODULE_DEPENDENCIES hpx_actions hpx_components_base hpx_config
                      hpx_parcelset_base
  CMAKE_SUBDIRS examples tests
)
