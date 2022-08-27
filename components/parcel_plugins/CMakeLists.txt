# Copyright (c) 2007-2022 Hartmut Kaiser
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(parcel_plugins ${parcel_plugins} binary_filter)

if(HPX_WITH_NETWORKING)
  set(parcel_plugins ${parcel_plugins} coalescing)
endif()

foreach(type ${parcel_plugins})
  add_hpx_pseudo_target(components.parcel_plugins.${type})
  add_subdirectory(${type})
  add_hpx_pseudo_dependencies(
    components.parcel_plugins components.parcel_plugins.${type}
  )
endforeach()
