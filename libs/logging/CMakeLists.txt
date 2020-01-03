# Copyright (c) 2019 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

cmake_minimum_required(VERSION 3.6.3 FATAL_ERROR)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# Default location is $HPX_ROOT/libs/logging/include
set(logging_headers
  hpx/logging.hpp
  hpx/logging/detail/time_format_holder.hpp
  hpx/logging/detail/macros.hpp
  hpx/logging/detail/manipulator.hpp
  hpx/logging/detail/logger.hpp
  hpx/logging/detail/format_write_detail.hpp
  hpx/logging/detail/cache_before_init.hpp
  hpx/logging/format.hpp
  hpx/logging/format/array.hpp
  hpx/logging/format/formatter/named_spacer.hpp
  hpx/logging/format/formatter/high_precision_time.hpp
  hpx/logging/format/formatter/defaults.hpp
  hpx/logging/format/formatter/thread_id.hpp
  hpx/logging/format/named_write.hpp
  hpx/logging/format/destination/named.hpp
  hpx/logging/format/destination/defaults.hpp
  hpx/logging/format/destination/file.hpp
  hpx/logging/level.hpp
  hpx/logging/logging.hpp
  hpx/logging/message.hpp
  hpx/logging/writer/format_write.hpp
  hpx/logging/writer/named_write.hpp
)


# Default location is $HPX_ROOT/libs/logging/include_compatibility
set(logging_compat_headers
  hpx/util/logging.hpp
  hpx/util/logging/format.hpp
  hpx/util/logging/writer/format_write.hpp
  hpx/util/logging/writer/named_write.hpp
  hpx/util/logging/detail/time_format_holder.hpp
  hpx/util/logging/detail/macros.hpp
  hpx/util/logging/detail/level.hpp
  hpx/util/logging/detail/manipulator.hpp
  hpx/util/logging/detail/logger.hpp
  hpx/util/logging/detail/format_write_detail.hpp
  hpx/util/logging/detail/cache_before_init.hpp
  hpx/util/logging/logging.hpp
  hpx/util/logging/format/formatter/named_spacer.hpp
  hpx/util/logging/format/formatter/high_precision_time.hpp
  hpx/util/logging/format/formatter/defaults.hpp
  hpx/util/logging/format/formatter/thread_id.hpp
  hpx/util/logging/format/array.hpp
  hpx/util/logging/format/named_write.hpp
  hpx/util/logging/format/destination/named.hpp
  hpx/util/logging/format/destination/defaults.hpp
  hpx/util/logging/format/destination/file.hpp
  hpx/util/logging/format/op_equal.hpp
  hpx/util/logging/format/optimize.hpp
)

# Default location is $HPX_ROOT/libs/logging/src
set(logging_sources
  level.cpp
  logging.cpp
  message.cpp
)

include(HPX_AddModule)
add_hpx_module(logging
  COMPATIBILITY_HEADERS ON
  DEPRECATION_WARNINGS
  FORCE_LINKING_GEN
  GLOBAL_HEADER_GEN OFF
  SOURCES ${logging_sources}
  HEADERS ${logging_headers}
  COMPAT_HEADERS ${logging_compat_headers}
  DEPENDENCIES
    hpx_assertion
    hpx_config
    hpx_format
    hpx_filesystem
    hpx_format
    hpx_timing
  CMAKE_SUBDIRS examples tests
)
