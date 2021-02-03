# Copyright (c) 2019 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

cmake_minimum_required(VERSION 3.13 FATAL_ERROR)

set(iterator_support_headers
    hpx/iterator_support/traits/is_iterator.hpp
    hpx/iterator_support/traits/is_range.hpp
    hpx/iterator_support/traits/is_sentinel_for.hpp
    hpx/iterator_support/counting_iterator.hpp
    hpx/iterator_support/generator_iterator.hpp
    hpx/iterator_support/iterator_adaptor.hpp
    hpx/iterator_support/iterator_facade.hpp
    hpx/iterator_support/iterator_range.hpp
    hpx/iterator_support/range.hpp
    hpx/iterator_support/transform_iterator.hpp
    hpx/iterator_support/zip_iterator.hpp
)

# cmake-format: off
set(iterator_support_compat_headers
    hpx/iterator_support.hpp => hpx/modules/iterator_support.hpp
    hpx/traits/is_iterator.hpp => hpx/modules/iterator_support.hpp
    hpx/traits/is_range.hpp => hpx/modules/iterator_support.hpp
    hpx/util/iterator_adaptor.hpp => hpx/modules/iterator_support.hpp
    hpx/util/iterator_facade.hpp => hpx/modules/iterator_support.hpp
    hpx/util/iterator_range.hpp => hpx/modules/iterator_support.hpp
    hpx/util/range.hpp => hpx/modules/iterator_support.hpp
    hpx/util/transform_iterator.hpp => hpx/modules/iterator_support.hpp
)
# cmake-format: on

set(iterator_support_sources)

include(HPX_AddModule)
add_hpx_module(
  core iterator_support
  COMPATIBILITY_HEADERS ON # Added in 1.5.0
  DEPRECATION_WARNINGS
  GLOBAL_HEADER_GEN ON
  EXCLUDE_FROM_GLOBAL_HEADER "hpx/iterator_support/traits/is_iterator.hpp"
                             "hpx/iterator_support/traits/is_range.hpp"
  SOURCES ${iterator_support_sources}
  HEADERS ${iterator_support_headers}
  COMPAT_HEADERS ${iterator_support_compat_headers}
  MODULE_DEPENDENCIES hpx_config hpx_datastructures hpx_functional
                      hpx_serialization hpx_type_support
  CMAKE_SUBDIRS examples tests
)
