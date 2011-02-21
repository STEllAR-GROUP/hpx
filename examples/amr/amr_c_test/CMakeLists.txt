# Copyright (c) 2007-2009 Hartmut Kaiser
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying 
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

###############################################################################
set(sources
    amr_c_test.cpp
    rand.cpp)

###############################################################################
source_group("Source Files" FILES ${sources})

###############################################################################
add_hpx_component(amr_c_test
  MODULE amr
  SOURCES ${sources}) 

