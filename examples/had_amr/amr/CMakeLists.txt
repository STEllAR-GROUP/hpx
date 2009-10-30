# Copyright (c) 2007-2009 Hartmut Kaiser
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying 
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

###############################################################################
set(had_amr_client_HEADERS
    amr_mesh.hpp
    amr_mesh_tapered.hpp
    functional_component.hpp
    dynamic_stencil_value.hpp)
source_group("Header Files" 
    FILES ${had_amr_client_HEADERS})

set(had_amr_stubs_HEADERS
    stubs/amr_mesh.hpp
    stubs/amr_mesh_tapered.hpp
    stubs/functional_component.hpp
    stubs/dynamic_stencil_value.hpp)
source_group("Header Files\\Stubs" 
    FILES ${had_amr_stubs_HEADERS})

set(had_amr_server_HEADERS
    server/amr_mesh.hpp
    server/amr_mesh_tapered.hpp
    server/functional_component.hpp
    server/stencil_value_in_adaptor.hpp
    server/stencil_value_out_adaptor.hpp
    server/dynamic_stencil_value.hpp
    server/dynamic_stencil_value_impl.hpp)
source_group("Header Files\\Server" 
    FILES ${had_amr_server_HEADERS})

set(had_amr_HEADERS
    ${had_amr_client_HEADERS}
    ${had_amr_stubs_HEADERS}
    ${had_amr_server_HEADERS})

###############################################################################
set(had_amr_client_SOURCES
    amr.cpp)
source_group("Source Files" 
    FILES ${had_amr_client_SOURCES})

set(had_amr_server_SOURCES
    server/amr_mesh.cpp
    server/amr_mesh_tapered.cpp
    server/functional_component.cpp
    server/stencil_value_out_adaptor.cpp
    server/dynamic_stencil_value.cpp)
source_group("Source Files\\Server" 
    FILES ${had_amr_server_SOURCES})

set(had_amr_SOURCES
    ${had_amr_client_SOURCES}
    ${had_amr_server_SOURCES})

###############################################################################
set(had_amr_INI
    had_amr.ini)
source_group("" 
    FILES ${had_amr_INI})

###############################################################################
add_hpx_component(had_amr 
    SOURCES ${had_amr_SOURCES} 
    HEADERS ${had_amr_HEADERS}
    INI ${had_amr_INI})

