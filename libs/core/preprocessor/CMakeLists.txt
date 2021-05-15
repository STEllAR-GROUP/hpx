# Copyright (c) 2019 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# Default location is $HPX_ROOT/libs/preprocessor/include
set(preprocessor_headers
    hpx/preprocessor/cat.hpp
    hpx/preprocessor/config.hpp
    hpx/preprocessor/expand.hpp
    hpx/preprocessor/identity.hpp
    hpx/preprocessor/nargs.hpp
    hpx/preprocessor/stringize.hpp
    hpx/preprocessor/strip_parens.hpp
)

# Default location is $HPX_ROOT/libs/preprocessor/include_compatibility
# cmake-format: off
set(preprocessor_compat_headers
    hpx/preprocessor.hpp => hpx/modules/preprocessor.hpp
)
# cmake-format: on

# Default location is $HPX_ROOT/libs/preprocessor/src
set(preprocessor_sources)

include(HPX_AddModule)
add_hpx_module(
  core preprocessor
  SOURCES ${preprocessor_sources}
  HEADERS ${preprocessor_headers}
  COMPAT_HEADERS ${preprocessor_compat_headers}
  CMAKE_SUBDIRS tests
)
