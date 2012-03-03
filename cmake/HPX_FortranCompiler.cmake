# Copyright (c) 2011 Bryce Lelbach
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(HPX_FORTRANCOMPILER_LOADED TRUE)

include(HPX_Include)

hpx_include(Message)

if(NOT HPX_FORTRAN_SEARCHED)
  include(CMakeDetermineFortranCompiler)

  if(CMAKE_Fortran_COMPILER)
    hpx_info("fortran" "Found a Fortran compiler")
    enable_language(Fortran)
  else()
    hpx_warn("fortran" "Couldn't find a Fortran compiler")
  endif()

  set(HPX_FORTRAN_SEARCHED ON CACHE INTERNAL "Searched for a Fortran compiler.")
elseif(CMAKE_Fortran_COMPILER)
  enable_language(Fortran)
endif()

