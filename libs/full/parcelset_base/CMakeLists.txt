# Copyright (c) 2019-2021 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(parcelset_base_headers
    hpx/parcelset_base/detail/locality_interface_functions.hpp
    hpx/parcelset_base/locality.hpp
    hpx/parcelset_base/parcelset_base_fwd.hpp
    hpx/parcelset_base/locality_interface.hpp
    hpx/parcelset_base/parcel_interface.hpp
    hpx/parcelset_base/policies/message_handler.hpp
    hpx/parcelset_base/traits/action_get_embedded_parcel.hpp
    hpx/parcelset_base/traits/action_message_handler.hpp
    hpx/parcelset_base/traits/action_serialization_filter.hpp
)

# cmake-format: off
set(parcelset_base_compat_headers
    hpx/runtime/parcelset/locality.hpp => hpx/modules/parcelset_base.hpp
    hpx/runtime/parcelset/policies/message_handler.hpp => hpx/modules/parcelset_base.hpp
    hpx/runtime/traits/action_get_embedded_parcel.hpp => hpx/modules/parcelset_base.hpp
    hpx/runtime/traits/action_serialization_filter.hpp => hpx/modules/parcelset_base.hpp
)
# cmake-format: on

set(parcelset_base_sources detail/locality_interface_functions.cpp locality.cpp
                           locality_interface.cpp parcel_interface.cpp
)

include(HPX_AddModule)
add_hpx_module(
  full parcelset_base
  GLOBAL_HEADER_GEN ON
  SOURCES ${parcelset_base_sources}
  HEADERS ${parcelset_base_headers}
  COMPAT_HEADERS ${parcelset_base_compat_headers}
  DEPENDENCIES HPX::hpx_local
  MODULE_DEPENDENCIES hpx_config hpx_naming_base
  CMAKE_SUBDIRS examples tests
)
