# Copyright (c) 2007-2022 Hartmut Kaiser
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(binary_filter_plugins)

if(HPX_WITH_NETWORKING)
  set(binary_filter_plugins ${binary_filter_plugins} bzip2 snappy zlib)
endif()

foreach(type ${binary_filter_plugins})
  add_hpx_pseudo_target(components.parcel_plugins.binary_filter.${type})
  add_subdirectory(${type})
  add_hpx_pseudo_dependencies(
    components.parcel_plugins.binary_filter
    components.parcel_plugins.binary_filter.${type}
  )
endforeach()
