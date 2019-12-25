# Copyright (c) 2019 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

cmake_minimum_required(VERSION 3.3.2 FATAL_ERROR)

# Default location is $HPX_ROOT/libs/lexical_cast/include
set(lexical_cast_headers
    hpx/lexical_cast/bad_lexical_cast.hpp
    hpx/lexical_cast/lexical_cast.hpp
    hpx/lexical_cast/safe_lexical_cast.hpp
    hpx/lexical_cast/try_lexical_convert.hpp
    hpx/lexical_cast/detail/basic_pointerbuf.hpp
    hpx/lexical_cast/detail/converter_lexical.hpp
    hpx/lexical_cast/detail/converter_lexical_streams.hpp
    hpx/lexical_cast/detail/converter_numeric.hpp
    hpx/lexical_cast/detail/cstring_wrapper.hpp
    hpx/lexical_cast/detail/inf_nan.hpp
    hpx/lexical_cast/detail/is_character.hpp
    hpx/lexical_cast/detail/lcast_char_constants.hpp
    hpx/lexical_cast/detail/lcast_precision.hpp
    hpx/lexical_cast/detail/lcast_unsigned_converters.hpp
)

# Default location is $HPX_ROOT/libs/lexical_cast/include_compatibility
set(lexical_cast_compat_headers
  hpx/util/safe_lexical_cast.hpp
)

# Default location is $HPX_ROOT/libs/lexical_cast/src
set(lexical_cast_sources
  bad_lexical_cast.cpp
)

include(HPX_AddModule)
add_hpx_module(lexical_cast
  COMPATIBILITY_HEADERS ON
  FORCE_LINKING_GEN
  GLOBAL_HEADER_GEN ON
  EXCLUDE_FROM_GLOBAL_HEADER
    hpx/lexical_cast/detail/basic_pointerbuf.hpp
    hpx/lexical_cast/detail/converter_lexical.hpp
    hpx/lexical_cast/detail/converter_lexical_streams.hpp
    hpx/lexical_cast/detail/converter_numeric.hpp
    hpx/lexical_cast/detail/cstring_wrapper.hpp
    hpx/lexical_cast/detail/inf_nan.hpp
    hpx/lexical_cast/detail/is_character.hpp
    hpx/lexical_cast/detail/lcast_char_constants.hpp
    hpx/lexical_cast/detail/lcast_precision.hpp
    hpx/lexical_cast/detail/lcast_unsigned_converters.hpp
  SOURCES ${lexical_cast_sources}
  HEADERS ${lexical_cast_headers}
  COMPAT_HEADERS ${lexical_cast_compat_headers}
  DEPENDENCIES
    hpx_config
    hpx_assertion
  CMAKE_SUBDIRS examples tests
)
