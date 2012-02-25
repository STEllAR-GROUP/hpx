# Copyright (c) 2011-2012 Bryce Adelstein-Lelbach
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

if(NOT HPX_FINDPROGRAM_LOADED)
  include(HPX_FindProgram)
endif()

if(NOT XSLTPROC_ROOT)
  if($ENV{XSLTPROC_ROOT})
    set(XSLTPROC_ROOT $ENV{XSLTPROC_ROOT})
  endif()
endif()

hpx_find_program(XSLTPROC
  PROGRAMS xsltproc
  PROGRAM_PATHS bin)

