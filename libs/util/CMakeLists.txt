# Copyright (c) 2019 The STE||AR-Group
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

cmake_minimum_required(VERSION 3.3.2 FATAL_ERROR)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# Default location is $HPX_ROOT/libs/util/include
set(util_headers
  hpx/util/detail/pack.hpp
  hpx/util/allocator_deleter.hpp
  hpx/util/assert_owns_lock.hpp
  hpx/util/atomic_count.hpp
  hpx/util/await_traits.hpp
  hpx/util/calculate_fanout.hpp
  hpx/util/fibhash.hpp
  hpx/util/get_and_reset_value.hpp
  hpx/util/insert_checked.hpp
  hpx/util/internal_allocator.hpp
  hpx/util/iterator_adaptor.hpp
  hpx/util/iterator_facade.hpp
  hpx/util/iterator_range.hpp
  hpx/util/jenkins_hash.hpp
  hpx/util/manage_config.hpp
  hpx/util/max.hpp
  hpx/util/memory_chunk.hpp
  hpx/util/min.hpp
  hpx/util/optional.hpp
  hpx/util/range.hpp
  hpx/util/safe_lexical_cast.hpp
  hpx/util/set_thread_name.hpp
  hpx/util/thread_specific_ptr.hpp
  hpx/util/unlock_guard.hpp
  hpx/traits/detail/reserve.hpp
  hpx/traits/concepts.hpp
  hpx/traits/has_member_xxx.hpp
  hpx/traits/has_xxx.hpp
  hpx/traits/is_iterator.hpp
  hpx/traits/is_range.hpp
)

# Default location is $HPX_ROOT/libs/util/src
set(util_sources
  manage_config.cpp
  set_thread_name.cpp
)

include(HPX_AddModule)
add_hpx_module(util
    COMPATIBILITY_HEADERS OFF
    DEPRECATION_WARNINGS
    FORCE_LINKING_GEN
    GLOBAL_HEADER_GEN ON
    EXCLUDE_FROM_GLOBAL_HEADER
      "traits/concepts.hpp"
      "traits/detail/reserve.hpp"
      "traits/has_member_xxx.hpp"
      "traits/has_xxx.hpp"
      "traits/is_iterator.hpp"
      "traits/is_range.hpp"
      "util/detail/pack.hpp"
    SOURCES ${util_sources}
    HEADERS ${util_headers}
    COMPAT_HEADERS ${util_compat_headers}
    DEPENDENCIES hpx_assertion hpx_config hpx_type_support
    CMAKE_SUBDIRS examples tests
)
