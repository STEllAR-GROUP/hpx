# Copyright (c) 2019 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# Default location is $HPX_ROOT/libs/config/include
set(config_headers
    hpx/local/config.hpp
    hpx/local/config/asio.hpp
    hpx/local/config/attributes.hpp
    hpx/local/config/autolink.hpp
    hpx/local/config/branch_hints.hpp
    hpx/local/config/compiler_fence.hpp
    hpx/local/config/compiler_native_tls.hpp
    hpx/local/config/compiler_specific.hpp
    hpx/local/config/constexpr.hpp
    hpx/local/config/debug.hpp
    hpx/local/config/deprecation.hpp
    hpx/local/config/detail/compat_error_code.hpp
    hpx/local/config/emulate_deleted.hpp
    hpx/local/config/endian.hpp
    hpx/local/config/export_definitions.hpp
    hpx/local/config/forceinline.hpp
    hpx/local/config/lambda_capture.hpp
    hpx/local/config/lambda_capture_this.hpp
    hpx/local/config/manual_profiling.hpp
    hpx/local/config/threads_stack.hpp
    hpx/local/config/warnings_prefix.hpp
    hpx/local/config/warnings_suffix.hpp
)

# Default location is $HPX_ROOT/libs/config/src
set(config_sources version.cpp)

include(HPXLocal_AddModule)
hpx_local_add_module(
  local config_local CONFIG_FILES
  GLOBAL_HEADER_GEN OFF
  SOURCES ${config_sources}
  HEADERS ${config_headers}
  MODULE_DEPENDENCIES hpx_preprocessor
  DEPENDENCIES hpx_local_dependencies_boost
  CMAKE_SUBDIRS examples tests
)
