# Copyright (c) 2019 The STE||AR-Group
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

cmake_minimum_required(VERSION 3.3.2 FATAL_ERROR)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# Default location is $HPX_ROOT/libs/config/include
set(config_headers
  hpx/config.hpp
  hpx/config/asio.hpp
  hpx/config/attributes.hpp
  hpx/config/autolink.hpp
  hpx/config/branch_hints.hpp
  hpx/config/compiler_fence.hpp
  hpx/config/compiler_native_tls.hpp
  hpx/config/compiler_specific.hpp
  hpx/config/constexpr.hpp
  hpx/config/debug.hpp
  hpx/config/emulate_deleted.hpp
  hpx/config/export_definitions.hpp
  hpx/config/force_linking.hpp
  hpx/config/forceinline.hpp
  hpx/config/lambda_capture.hpp
  hpx/config/manual_profiling.hpp
  hpx/config/threads_stack.hpp
  hpx/config/warnings_prefix.hpp
  hpx/config/warnings_suffix.hpp
  hpx/config/weak_symbol.hpp
)

# Default location is $HPX_ROOT/libs/config/src
set(config_sources
  force_linking.cpp
  version.cpp
)

# Apex depends on hpx_config, need to export it
if(HPX_WITH_APEX)
  set(_export_module INSTALL_BINARIES ON EXPORT ON)
endif()

include(HPX_AddModule)
add_hpx_module(config
    COMPATIBILITY_HEADERS OFF
    GLOBAL_HEADER_GEN OFF
    FORCE_LINKING_GEN OFF
    ${_export_module}           # Necessary for apex
    SOURCES ${config_sources}
    HEADERS ${config_headers}
    DEPENDENCIES hpx_preprocessor
    CMAKE_SUBDIRS examples tests
)
