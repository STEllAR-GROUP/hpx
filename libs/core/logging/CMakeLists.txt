# Copyright (c) 2019-2025 The STE||AR-Group
# Copyright (c) 2019-2025 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# Compatibility with various Boost types, introduced in V1.4.0
hpx_option(
  HPX_LOGGING_WITH_SEPARATE_DESTINATIONS
  BOOL
  "Enable separate logging channels for AGAS, timing, and parcel transport. (default: ON)"
  ON
  ADVANCED
  CATEGORY "Modules"
  MODULE LOGGING
)

if(HPX_LOGGING_WITH_SEPARATE_DESTINATIONS)
  hpx_add_config_define_namespace(
    DEFINE HPX_LOGGING_HAVE_SEPARATE_DESTINATIONS NAMESPACE LOGGING
  )
endif()

# Default location is $HPX_ROOT/libs/logging/include
set(logging_headers
    hpx/logging/api.hpp
    hpx/logging/detail/macros.hpp
    hpx/logging/detail/logger.hpp
    hpx/logging/format/destinations.hpp
    hpx/logging/format/formatters.hpp
    hpx/logging/format/named_write.hpp
    hpx/logging/level.hpp
    hpx/logging/logging.hpp
    hpx/logging/macros.hpp
    hpx/logging/manipulator.hpp
    hpx/logging/message.hpp
)
set(logging_macro_headers hpx/logging/detail/macros.hpp hpx/logging/macros.hpp)

# Default location is $HPX_ROOT/libs/logging/include_compatibility
# cmake-format: off
set(logging_compat_headers
    hpx/logging.hpp => hpx/modules/logging.hpp
    hpx/util/logging.hpp => hpx/modules/logging.hpp
    hpx/util/logging/writer/format_write.hpp => hpx/modules/logging.hpp
    hpx/util/logging/writer/named_write.hpp => hpx/modules/logging.hpp
    hpx/util/logging/logging.hpp => hpx/modules/logging.hpp
    hpx/util/logging/format/formatter/high_precision_time.hpp => hpx/modules/logging.hpp
    hpx/util/logging/format/formatter/defaults.hpp => hpx/modules/logging.hpp
    hpx/util/logging/format/formatter/thread_id.hpp => hpx/modules/logging.hpp
    hpx/util/logging/format/named_write.hpp => hpx/modules/logging.hpp
    hpx/util/logging/format/destination/defaults.hpp => hpx/modules/logging.hpp
    hpx/util/logging/format/destination/file.hpp => hpx/modules/logging.hpp
    hpx/util/logging/format/optimize.hpp => hpx/modules/logging.hpp
)
# cmake-format: on

# Default location is $HPX_ROOT/libs/logging/src
set(logging_sources
    level.cpp
    logging.cpp
    manipulator.cpp
    format/named_write.cpp
    format/destination/defaults_destination.cpp
    format/destination/file.cpp
    format/formatter/high_precision_time.cpp
    format/formatter/defaults_formatter.cpp
    format/formatter/thread_id.cpp
)

include(HPX_AddModule)
add_hpx_module(
  core logging
  GLOBAL_HEADER_GEN ON
  GLOBAL_HEADER_MODULE_GEN ON
  SOURCES ${logging_sources}
  HEADERS ${logging_headers}
  MACRO_HEADERS ${logging_macro_headers}
  COMPAT_HEADERS ${logging_compat_headers}
  MODULE_DEPENDENCIES
    hpx_assertion
    hpx_config
    hpx_format
    hpx_filesystem
    hpx_format
    hpx_preprocessor
    hpx_timing
    hpx_thread_support
    hpx_type_support
  CMAKE_SUBDIRS examples tests
)
