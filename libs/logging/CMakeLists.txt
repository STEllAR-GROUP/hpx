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
  hpx/logging/detail/macros.hpp
  hpx/logging/detail/logger.hpp
  hpx/logging/format/destinations.hpp
  hpx/logging/format/formatters.hpp
  hpx/logging/format/named_write.hpp
  hpx/logging/level.hpp
  hpx/logging/logging.hpp
  hpx/logging/manipulator.hpp
  hpx/logging/message.hpp
)

# Default location is $HPX_ROOT/libs/logging/include_compatibility
set(logging_compat_headers
  hpx/util/logging.hpp
  hpx/util/logging/writer/format_write.hpp
  hpx/util/logging/writer/named_write.hpp
  hpx/util/logging/detail/macros.hpp
  hpx/util/logging/detail/level.hpp
  hpx/util/logging/detail/manipulator.hpp
  hpx/util/logging/detail/logger.hpp
  hpx/util/logging/logging.hpp
  hpx/util/logging/format/formatter/high_precision_time.hpp
  hpx/util/logging/format/formatter/defaults.hpp
  hpx/util/logging/format/formatter/thread_id.hpp
  hpx/util/logging/format/named_write.hpp
  hpx/util/logging/format/destination/defaults.hpp
  hpx/util/logging/format/destination/file.hpp
  hpx/util/logging/format/optimize.hpp
)

# Default location is $HPX_ROOT/libs/logging/src
set(logging_sources
  level.cpp
  logging.cpp
  manipulator.cpp
  format/named_write.cpp
  format/destination/defaults.cpp
  format/destination/file.cpp
  format/formatter/high_precision_time.cpp
  format/formatter/defaults.cpp
  format/formatter/thread_id.cpp
  format/named_write.cpp
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
