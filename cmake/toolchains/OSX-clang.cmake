# Copyright (c) 2014 Thomas Heller
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

# This is the default toolchain file to be used with OSX building HPX with clang. It sets
# the appropriate compile flags and compiler such that HPX will compile.
# Note that you still need to provide Boost, hwloc and other utility libraries
# like a custom allocator yourself.

# Set the Clang Compiler
set(CMAKE_CXX_COMPILER clang++)
set(CMAKE_C_COMPILER clang)
#set(CMAKE_Fortran_COMPILER)

# Add the -stdlib=libc++ compile flag such that everything will be compiled for the correct
# platform
set(CMAKE_CXX_FLAGS_INIT "-stdlib=libc++" CACHE STRING "Initial compiler flags used to compile for OSX")

# HWLOC isn't working on OSX
set(HPX_WITH_HWLOC OFF CACHE BOOL
  "Use Hwloc for hardware topolgy information and thread pinning. If disabled, performance might be reduced.")

# APPLEs clang doesn't know how to deal with native tls properly
set(HPX_NATIVE_TLS OFF CACHE BOOL "Use native TLS support if available (default: ON)")

# Clang doesn't know about hidden visibility
set(HPX_HIDDEN_VISIBILITY OFF CACHE BOOL
  "Use -fvisibility=hidden for builds on platforms which support it (default ON)")

# We don't do cross compilation here ...
set(CMAKE_CROSSCOMPILING OFF)
