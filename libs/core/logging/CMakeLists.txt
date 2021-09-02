# Copyright (c) 2019 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# Default location is $HPX_ROOT/libs/logging/include
set(logging_headers
    hpx/modules/logging.hpp
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

include(HPXLocal_AddModule)
hpx_local_add_module(
  local logging
  GLOBAL_HEADER_GEN OFF
  SOURCES ${logging_sources}
  HEADERS ${logging_headers}
  COMPAT_HEADERS ${logging_compat_headers}
  MODULE_DEPENDENCIES
    hpx_assertion
    hpx_config_local
    hpx_format
    hpx_filesystem
    hpx_format
    hpx_timing
    hpx_thread_support
  CMAKE_SUBDIRS examples tests
)
