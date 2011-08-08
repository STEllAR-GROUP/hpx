# Copyright (c) 2007-2009 Hartmut Kaiser
# Copyright (c) 2011 Bryce Lelbach
# Copyright (c) 2011 Vinay C Amatya
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying 
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

add_hpx_component(board
  MODULE nqueen
  SOURCES board.cpp)

if(MSVC)  
  set_property(TARGET board_component APPEND
    PROPERTY COMPILE_FLAGS
    "-bigobj")
endif()
 
add_hpx_executable(nqueen_client
  MODULE nqueen
  SOURCES nqueen_client.cpp
  DEPENDENCIES board_component)
   
# add dependencies to pseudo-target
add_hpx_pseudo_dependencies(examples.nqueen nqueen_client_exe)

