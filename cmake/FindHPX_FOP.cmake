# Copyright (c) 2011-2012 Bryce Adelstein-Lelbach
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

if(NOT HPX_FINDPROGRAM_LOADED)
  include(HPX_FindProgram)
endif()

if(NOT FOP_ROOT)
  if($ENV{FOP_ROOT})
    set(FOP_ROOT $ENV{FOP_ROOT})
  endif()
endif()

hpx_find_program(FOP
  PROGRAMS fop
  PROGRAM_PATHS . bin)

