# Copyright (c) 2019 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

cmake_minimum_required(VERSION 3.13 FATAL_ERROR)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# Default location is $HPX_ROOT/libs/program_options/include
set(program_options_headers
    hpx/program_options.hpp
    hpx/modules/program_options.hpp
    hpx/program_options/cmdline.hpp
    hpx/program_options/config.hpp
    hpx/program_options/environment_iterator.hpp
    hpx/program_options/eof_iterator.hpp
    hpx/program_options/errors.hpp
    hpx/program_options/option.hpp
    hpx/program_options/options_description.hpp
    hpx/program_options/parsers.hpp
    hpx/program_options/positional_options.hpp
    hpx/program_options/value_semantic.hpp
    hpx/program_options/variables_map.hpp
    hpx/program_options/version.hpp
    hpx/program_options/detail/cmdline.hpp
    hpx/program_options/detail/config_file.hpp
    hpx/program_options/detail/convert.hpp
    hpx/program_options/detail/parsers.hpp
    hpx/program_options/detail/utf8_codecvt_facet.hpp
    hpx/program_options/detail/value_semantic.hpp
)

# cmake-format: off
set(program_options_compat_headers)
# cmake-format: on

# Default location is $HPX_ROOT/libs/program_options/src
set(program_options_sources
    cmdline.cpp
    config_file.cpp
    convert.cpp
    options_description.cpp
    parsers.cpp
    positional_options.cpp
    split.cpp
    utf8_codecvt_facet.cpp
    value_semantic.cpp
    variables_map.cpp
    winmain.cpp
)

include(HPX_AddModule)
add_hpx_module(
  full program_options
  GLOBAL_HEADER_GEN OFF
  SOURCES ${program_options_sources}
  HEADERS ${program_options_headers}
  COMPAT_HEADERS ${program_options_compat_headers}
  DEPENDENCIES hpx_core hpx_parallelism ${__boost_program_options}
  CMAKE_SUBDIRS examples tests
)
