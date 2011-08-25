# Copyright (c) 2007-2009 Hartmut Kaiser
# Copyright (c) 2011 Matt Anderson
# Copyright (c) 2011 Bryce Lelbach
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying 
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

###############################################################################
set(sources
    had_amr3d_c_test.cpp)

###############################################################################
source_group("Source Files" FILES ${sources})

###############################################################################
add_hpx_library(ssmp_amr3d_c_test
  MODULE ssmp_amr3d
  SOURCES ${sources})

