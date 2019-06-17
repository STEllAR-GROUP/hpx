# Copyright (c) 2019 The STE||AR-Group
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

cmake_minimum_required(VERSION 3.3.2 FATAL_ERROR)

set(module_name preprocessor)
string(TOUPPER ${module_name} upper_module_name)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

project(HPX.${module_name} CXX)

# Default location is $HPX_ROOT/libs/${module_name}/include
set(${module_name}_headers
  hpx/preprocessor.hpp
  hpx/${module_name}/cat.hpp
  hpx/${module_name}/config.hpp
  hpx/${module_name}/expand.hpp
  hpx/${module_name}/nargs.hpp
  hpx/${module_name}/stringize.hpp
  hpx/${module_name}/strip_parens.hpp
)

# Default location is $HPX_ROOT/libs/${module_name}/include_compatibility
if(HPX_${upper_module_name}_WITH_COMPATIBILITY_HEADERS)
  set(${module_name}_compat_headers
    hpx/util/detail/pp/cat.hpp
    hpx/util/detail/pp/config.hpp
    hpx/util/detail/pp/expand.hpp
    hpx/util/detail/pp/nargs.hpp
    hpx/util/detail/pp/stringize.hpp
    hpx/util/detail/pp/strip_parens.hpp
  )
endif()

# Default location is $HPX_ROOT/libs/${module_name}/src
set(${module_name}_sources
  preprocessor.cpp
)

include(HPX_AddModule)
add_hpx_module(${module_name}
    DEPRECATION_WARNINGS
    COMPATIBILITY_HEADERS OFF
    SOURCES ${${module_name}_sources}
    HEADERS ${${module_name}_headers}
    COMPAT_HEADERS ${${module_name}_compat_headers}
    DEPENDENCIES hpx_config
    CMAKE_SUBDIRS src tests
)
