# Copyright (c) 2011 Bryce Lelbach
# Copyright (c) 2012 Thomas Heller
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

if(NOT HPX_FINDPROGRAM_LOADED)
  include(HPX_FindProgram)
endif()

if(NOT BOOSTWAVE_ROOT)
  if(BOOST_ROOT)
      set(BOOSTWAVE_ROOT ${BOOST_ROOT})
  elseif($ENV{BOOST_ROOT})
      set(BOOSTWAVE_ROOT $ENV{BOOST_ROOT})
  endif()
endif()

hpx_find_program(BOOSTWAVE
  PROGRAMS wave
  PROGRAM_PATHS bin dist/bin)
