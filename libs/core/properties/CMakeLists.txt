# Copyright (c) 2019-2021 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(properties_headers
    hpx/properties/is_applicable_property.hpp hpx/properties/prefer.hpp
    hpx/properties/query.hpp hpx/properties/require.hpp
    hpx/properties/require_concept.hpp hpx/properties/static_query.hpp
)

include(HPX_AddModule)
add_hpx_module(
  core properties
  GLOBAL_HEADER_GEN ON
  HEADERS ${properties_headers}
  DEPENDENCIES hpx_functional
  CMAKE_SUBDIRS examples tests
)
