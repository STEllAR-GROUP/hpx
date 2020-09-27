# Copyright (c) 2019 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

cmake_minimum_required(VERSION 3.13 FATAL_ERROR)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# Default location is $HPX_ROOT/libs/errors/include
set(errors_headers
    hpx/errors/error.hpp
    hpx/errors/error_code.hpp
    hpx/errors/exception.hpp
    hpx/errors/exception_fwd.hpp
    hpx/errors/exception_info.hpp
    hpx/errors/exception_list.hpp
    hpx/errors/throw_exception.hpp
    hpx/exception.hpp
    hpx/system_error.hpp
)

# Default location is $HPX_ROOT/libs/errors/include_compatibility
# cmake-format: off
set(errors_compat_headers
    hpx/error.hpp => hpx/system_error.hpp
    hpx/error_code.hpp => hpx/system_error.hpp
    hpx/errors.hpp => hpx/modules/errors.hpp
    hpx/exception_fwd.hpp => hpx/exception.hpp
    hpx/exception_info.hpp => hpx/modules/errors.hpp
    hpx/exception_list.hpp => hpx/modules/errors.hpp
    hpx/throw_exception.hpp => hpx/exception.hpp
)
# cmake-format: on

# Default location is $HPX_ROOT/libs/errors/src
set(errors_sources error_code.cpp exception.cpp exception_list.cpp
                   throw_exception.cpp
)

include(HPX_AddModule)
add_hpx_module(
  core errors
  COMPATIBILITY_HEADERS ON # Added in 1.5.0
  DEPRECATION_WARNINGS
  GLOBAL_HEADER_GEN ON
  SOURCES ${errors_sources}
  HEADERS ${errors_headers}
  COMPAT_HEADERS ${errors_compat_headers}
  MODULE_DEPENDENCIES
    hpx_assertion
    hpx_config
    hpx_filesystem
    hpx_format
    hpx_logging
    hpx_preprocessor
    hpx_thread_support
  CMAKE_SUBDIRS examples tests
)
