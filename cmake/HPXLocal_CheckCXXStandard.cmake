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
hpx_local_option(
  HPXLocal_CXX_STANDARD STRING
  "C++ standard to use for compiling HPX (default: 17)" "17" ADVANCED
)

if(HPXLocal_CXX_STANDARD LESS 17)
  hpx_local_error(
    "You've set HPXLocal_CXX_STANDARD to ${HPXLocal_CXX_STANDARD}, which is less than 17 which is the minimum required by HPX"
  )
else()
  set(CMAKE_CXX_STANDARD
      ${HPXLocal_CXX_STANDARD}
      CACHE INTERNAL "" FORCE
  )
endif()

set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_EXTENSIONS OFF)
# We explicitly set the default to 98 to force CMake to emit a -std=c++XX flag.
# Some compilers (clang) have a different default standard for cpp and cu files,
# but CMake does not know about this difference. If the standard is set to the
# .cpp default in CMake, CMake will omit the flag, resulting in the wrong
# standard for .cu files.
set(CMAKE_CXX_STANDARD_DEFAULT 98)

hpx_local_info("Using C++${HPXLocal_CXX_STANDARD}")
