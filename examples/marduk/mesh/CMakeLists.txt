# Copyright (c) 2007-2009 Hartmut Kaiser
# Copyright (c)      2011 Matthew Anderson
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying 
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

###############################################################################
set(client_sources
    amr.cpp)

set(server_sources
    server/unigrid_mesh.cpp
    server/functional_component.cpp
    server/stencil_value_out_adaptor.cpp
    server/dynamic_stencil_value.cpp)

###############################################################################
set(client_headers
    unigrid_mesh.hpp
    functional_component.hpp
    dynamic_stencil_value.hpp)

set(stubs_headers
    stubs/unigrid_mesh.hpp
    stubs/functional_component.hpp
    stubs/dynamic_stencil_value.hpp)

set(server_headers
    server/unigrid_mesh.hpp
    server/functional_component.hpp
    server/stencil_value_in_adaptor.hpp
    server/stencil_value_out_adaptor.hpp
    server/dynamic_stencil_value.hpp
    server/dynamic_stencil_value_impl.hpp)

###############################################################################
set(sources
    ${client_sources}
    ${server_sources})

set(headers
    ${client_headers}
    ${stubs_headers}
    ${server_headers})

###############################################################################
source_group("Source Files\\Client" FILES ${client_sources})
source_group("Source Files\\Server" FILES ${server_sources})
source_group("Header Files\\Client" FILES ${client_headers})
source_group("Header Files\\Stubs"  FILES ${stubs_headers})
source_group("Header Files\\Server" FILES ${server_headers})

###############################################################################
add_hpx_component(marduk_mesh 
  MODULE marduk_mesh
  SOURCES ${sources} 
  HEADERS ${headers})

