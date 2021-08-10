# Copyright (c) 2020 Mikael Simberg
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

# We require at least C++17. However, if a higher standard is set by the user in
# CMAKE_CXX_STANDARD that requirement has to be propagated to users of HPX as
# well (i.e. HPX can't be compiled with C++20 and applications with C++17; the
# other way around is allowed). Ideally, users should not set CMAKE_CXX_STANDARD
# when building HPX.
set(HPX_CXX_STANDARD 17)

# Compatibility for old HPX_WITH_CXXAB options.
if(HPX_WITH_CXX11)
  hpx_error(
    "HPX_WITH_CXX11 is deprecated and the minimum C++ standard required by HPX is 17. Avoid setting the standard explicitly or use CMAKE_CXX_STANDARD and HPX_USE_CMAKE_CXX_STANDARD if you must set it."
  )
elseif(HPX_WITH_CXX14)
  hpx_error(
    "HPX_WITH_CXX14 is deprecated and the minimum C++ standard required by HPX is 17. Avoid setting the standard explicitly or use CMAKE_CXX_STANDARD and HPX_USE_CMAKE_CXX_STANDARD if you must set it."
  )
elseif(HPX_WITH_CXX17)
  hpx_warn(
    "HPX_WITH_CXX17 is deprecated. Avoid setting the standard explicitly or use CMAKE_CXX_STANDARD and HPX_USE_CMAKE_CXX_STANDARD if you must set it."
  )
  set(HPX_CXX_STANDARD 17)
elseif(HPX_WITH_CXX20)
  hpx_warn(
    "HPX_WITH_CXX20 is deprecated. Avoid setting the standard explicitly or use CMAKE_CXX_STANDARD and HPX_USE_CMAKE_CXX_STANDARD if you must set it."
  )
  set(HPX_CXX_STANDARD 20)
endif()

if(CMAKE_CXX_STANDARD)
  if(CMAKE_CXX_STANDARD LESS 17)
    hpx_error(
      "You've set CMAKE_CXX_STANDARD to ${CMAKE_CXX_STANDARD}, which is less than 17 which is the minimum required by HPX"
    )
  else()
    if(HPX_USE_CMAKE_CXX_STANDARD)
      hpx_warn(
        "You've set CMAKE_CXX_STANDARD manually, which is not recommended. However, HPX_USE_CMAKE_CXX_STANDARD=ON so we're taking it into account."
      )
      set(HPX_CXX_STANDARD ${CMAKE_CXX_STANDARD})
    else()
      hpx_error(
        "You've set CMAKE_CXX_STANDARD manually, which is not recommended. If you really want to set CMAKE_CXX_STANDARD, set HPX_USE_CMAKE_CXX_STANDARD=ON."
      )
    endif()
  endif()
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
