# Copyright (c) 2011 Matt Anderson
# Copyright (c) 2007-2012 Hartmut Kaiser
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

# We have to compile the Fortran sources into a separate static library as
# cmake does not support generating mixed C++/Fortran modules for MSVC.
add_hpx_library(tsf_fortran
  STATIC
  SOURCES server/setup.F90
  FOLDER "Examples/Applications/Fortran Threadsafe")

add_hpx_component(tsf
  SOURCES ${sources}
  DEPENDENCIES iostreams_component tsf_fortran_lib
  FOLDER "Examples/Applications/Fortran Threadsafe"
  AUTOGLOB)

