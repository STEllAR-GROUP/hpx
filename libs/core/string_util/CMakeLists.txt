# Copyright (c) 2019-2022 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(string_util_headers
    hpx/string_util/case_conv.hpp hpx/string_util/classification.hpp
    hpx/string_util/getenv.hpp hpx/string_util/split.hpp
    hpx/string_util/strerror.hpp hpx/string_util/trim.hpp
)

# Default location is $HPX_ROOT/libs/string_util/src
set(string_util_sources getenv.cpp strerror.cpp)

if(HPX_WITH_CXX20_MODULES)
  set(string_util_modulefiles hpx.core.string_util.cppm)
endif()

include(HPX_AddModule)
add_hpx_module(
  core string_util
  GLOBAL_HEADER_GEN MODULE GENERATE_EXPORT_DEFINITIONS
  HEADERS ${string_util_headers}
  SOURCES ${string_util_sources}
  MODULES ${string_util_modulefiles}
  MODULE_DEPENDENCIES hpx_config
  CMAKE_SUBDIRS examples tests
)
