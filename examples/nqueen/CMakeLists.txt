# Copyright (c) 2007-2009 Hartmut Kaiser
# Copyright (c) 2011 Bryce Lelbach
# Copyright (c) 2011 Vinay C Amatya
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying 
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(client_sources nqueen.cpp)

set(client_headers nqueen.hpp)

set(stubs_headers stubs/nqueen.hpp)

set(server_headers server/nqueen.hpp)

set(sources nqueen_client.cpp)

set(headers
   ${client_headers}
   ${stubs_headers}
   ${server_headers})

source_group("Source Files\\Client" FILES ${client_sources})
source_group("Header Files\\Client" FILES ${client_headers})
source_group("Header Files\\Client" FILES ${stubs_sources})
source_group("Header Files\\Client" FILES ${server_sources})
source_group("Source Files\\Client" FILES ${sources})


add_hpx_component(nqueen
  MODULE nqueen
  SOURCES ${client_sources}
  HEADERS ${headers})

if(MSVC)  
  set_property(TARGET nqueen_component APPEND
    PROPERTY COMPILE_FLAGS
    "-bigobj")
endif()
 
add_hpx_executable(nqueen_client
  MODULE nqueen
  SOURCES ${sources}
  DEPENDENCIES nqueen_component)
   
# add dependencies to pseudo-target
add_hpx_pseudo_dependencies(examples.nqueen nqueen_client_exe)

