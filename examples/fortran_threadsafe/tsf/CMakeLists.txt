# Copyright (c) 2011 Matt Anderson
# Copyright (c) 2007-2012 Hartmut Kaiser
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

# We have to compile the Fortran sources into a separate static library as
# cmake does not support generating mixed C++/Fortran modules for MSVC.
add_hpx_library(tsf_fortran
  STATIC                # will be ignored on Linux
  SOURCES server/setup_pl.F90 server/point_pl.cpp
  FOLDER "Examples/Applications/Fortran Threadsafe")

#add_custom_command(OUTPUT point_pl.cpp setup_pl.F90
#                   COMMAND threadsafe.pl 12
#                  )

# Special hackery to make everything work for Windows Intel compiler
if(WIN32 AND ${CMAKE_Fortran_COMPILER_ID} STREQUAL "Intel")
  # Make sure all Fortran symbols are lower case for Windows Intel ifort compiler.
  set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} /names:lowercase")
  if("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
    # Make sure we use the proper druntime libraries in Debug build.
    set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} /dbglibs")
  endif()
  # Make sure the C++ linker will find the Fortran runtime libraries.
  link_directories("$ENV{IFORT_COMPILER12}/compiler/lib/intel64")
endif()

add_hpx_component(tsf
  SOURCES ${sources}
  DEPENDENCIES iostreams_component tsf_fortran_lib
  FOLDER "Examples/Applications/Fortran Threadsafe"
  AUTOGLOB)

