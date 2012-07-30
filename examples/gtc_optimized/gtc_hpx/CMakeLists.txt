# Copyright (c) 2011 Matt Anderson
# Copyright (c) 2007-2012 Hartmut Kaiser
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

# We have to compile the Fortran sources into a separate static library as
# cmake does not support generating mixed C++/Fortran modules for MSVC.
add_hpx_library(gtc_hpx_fortran
  AUTOGLOB
  STATIC
  SOURCE_GLOB "*.F90"
  HEADER_GLOB ""
  FOLDER "Examples/Applications/GTC Optimized")

# Make sure all Fortran symbols are lower case for Windows Intel ifort compiler
if(WIN32 AND ${CMAKE_Fortran_COMPILER_ID} STREQUAL "Intel")
  set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} /names:lowercase")
  set(FORTRAN_LIBRARY_PATH, "C:/Program Files (x86)/Intel/Composer XE 2011 SP1")
  link_directories("${FORTRAN_LIBRARY_PATH}/compiler/lib/intel64")
endif()

add_hpx_component(gtc_hpx
  SOURCES ${sources}
  SOURCE_GLOB "*.cpp"
  HEADER_GLOB "*.hpp"
  DEPENDENCIES iostreams_component gtc_hpx_fortran_lib
  FOLDER "Examples/Applications/GTC Optimized"
  AUTOGLOB)

