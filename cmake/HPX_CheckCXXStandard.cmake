# Copyright (c) 2020 Mikael Simberg
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

# We require at least C++17. However, if a higher standard is set by the user in
# HPX_WITH_CXX_STANDARD (or CMAKE_CXX_STANDARD in conjunction with
# HPX_USE_CMAKE_CXX_STANDARD=ON) that requirement has to be propagated to users
# of HPX as well (i.e. HPX can't be compiled with C++20 and applications with
# C++17; the other way around is allowed). Ideally, users should not set
# CMAKE_CXX_STANDARD when building HPX.

set(HPX_CXX_STANDARD_DEFAULT 17)
if(CMAKE_CXX_STANDARD AND HPX_USE_CMAKE_CXX_STANDARD)
  set(HPX_CXX_STANDARD_DEFAULT ${CMAKE_CXX_STANDARD})
endif()

hpx_option(
  HPX_WITH_CXX_STANDARD
  STRING
  "Set the C++ standard to use when compiling HPX itself. (default: ${HPX_CXX_STANDARD_DEFAULT})"
  "${HPX_CXX_STANDARD_DEFAULT}"
)

set(HPX_CXX_STANDARD ${HPX_WITH_CXX_STANDARD})

# Compatibility for old HPX_WITH_CXXAB options.
if(HPX_WITH_CXX11)
  hpx_error(
    "HPX_WITH_CXX11 is deprecated and the minimum C++ standard required by HPX is 17. Use HPX_WITH_CXX_STANDARD instead."
  )
elseif(HPX_WITH_CXX14)
  hpx_error(
    "HPX_WITH_CXX14 is deprecated and the minimum C++ standard required by HPX is 17. Use HPX_WITH_CXX_STANDARD instead."
  )
elseif(HPX_WITH_CXX17)
  hpx_warn(
    "HPX_WITH_CXX17 is deprecated. Use HPX_WITH_CXX_STANDARD=17 instead."
  )
  set(HPX_CXX_STANDARD 17)
elseif(HPX_WITH_CXX20)
  hpx_warn(
    "HPX_WITH_CXX20 is deprecated. Use HPX_WITH_CXX_STANDARD=20 instead."
  )
  set(HPX_CXX_STANDARD 20)
elseif(HPX_WITH_CXX23)
  hpx_warn(
    "HPX_WITH_CXX23 is deprecated. Use HPX_WITH_CXX_STANDARD=23 instead."
  )
  set(HPX_CXX_STANDARD 23)
endif()

if(CMAKE_CXX_STANDARD)
  if(CMAKE_CXX_STANDARD LESS 17)
    hpx_error(
      "You've set CMAKE_CXX_STANDARD to ${CMAKE_CXX_STANDARD}, which is less than 17 which is the minimum required by HPX"
    )
  elseif(NOT HPX_USE_CMAKE_CXX_STANDARD)
    hpx_error(
      "You've set CMAKE_CXX_STANDARD manually, which is not recommended. Please set HPX_WITH_CXX_STANDARD instead."
    )
  endif()
  set(HPX_CXX_STANDARD ${CMAKE_CXX_STANDARD})
endif()

set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
# We explicitly set the default to 98 to force CMake to emit a -std=c++XX flag.
# Some compilers (clang) have a different default standard for cpp and cu files,
# but CMake does not know about this difference. If the standard is set to the
# .cpp default in CMake, CMake will omit the flag, resulting in the wrong
# standard for .cu files.
set(CMAKE_CXX_STANDARD_DEFAULT 98)

hpx_info("Using C++${HPX_CXX_STANDARD}")
