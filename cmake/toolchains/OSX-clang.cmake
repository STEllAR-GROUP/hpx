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
set(WITH_HWLOC OFF)

# APPLEs clang doesn't know how to deal with native tls properly
set(WITH_NATIVE_TLS OFF)

# We don't do cross compilation here ...
set(CMAKE_CROSSCOMPILING OFF)
