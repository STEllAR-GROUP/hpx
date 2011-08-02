# Copyright (c) 2007-2009 Hartmut Kaiser
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying 
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

###############################################################################
set(sources
    application.cpp)

###############################################################################
source_group("Source Files" FILES ${sources})

###############################################################################
add_hpx_library(adaptive1d_application
  MODULE adaptive1d_application
  SOURCES ${sources}) 

