# Copyright (c) 2011 Matt Anderson
# Copyright (c) 2007-2012 Hartmut Kaiser
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

# We have to compile the Fortran sources into a separate static library as
# cmake does not support generating mixed C++/Fortran modules for MSVC.
if(MSVC)
  set(CMAKE_Fortran_FLAGS "-fdefault-real-8 ${CMAKE_Fortran_FLAGS}")
  add_hpx_library(nekbone_hpx_fortran
    AUTOGLOB
    STATIC
    SOURCE_GLOB "*.F90" "*.f90" "*.f"
    HEADER_GLOB ""
    FOLDER "Examples/Applications/GTCX")
else()
  set(CMAKE_Fortran_FLAGS "-fdefault-real-8 ${CMAKE_Fortran_FLAGS}")
  add_hpx_library(nekbone_hpx_fortran
    AUTOGLOB
    SOURCE_GLOB "*.F90" "*.f90" "*.f" "*.cpp" "*.c"
    HEADER_GLOB ""
    FOLDER "Examples/Applications/GTCX")
endif()

# Special hackery to make everything work for Windows Intel compiler
if(MSVC AND ${CMAKE_Fortran_COMPILER_ID} STREQUAL "Intel")
  # Make sure all Fortran symbols are lower case for Windows Intel ifort compiler.
  set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} /names:lowercase")
  if("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
    # Make sure we use the proper runtime libraries in Debug build.
    set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} /dbglibs /debug:full /Od")
  endif()
  # Make sure the C++ linker will find the Fortran runtime libraries.
  link_directories("$ENV{IFORT_COMPILER12}/compiler/lib/intel64")
endif()

add_hpx_component(nekbone_hpx
  SOURCE_GLOB "*.cpp"
  HEADER_GLOB "*.hpp"
  DEPENDENCIES iostreams_component nekbone_hpx_fortran_lib
  FOLDER "Examples/Applications/GTCX"
  AUTOGLOB)

