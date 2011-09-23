# Copyright (c) 2007-2011 Hartmut Kaiser
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying 
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

###############################################################################
# add component
add_hpx_component(sine 
  MODULE sine
  SOURCES sine.cpp server/sine.cpp 
  HEADERS server/sine.cpp)

###############################################################################
# add example executable
add_hpx_executable(sine_client 
  MODULE sine_client
  SOURCES sine_client.cpp 
  DEPENDENCIES sine_component)

