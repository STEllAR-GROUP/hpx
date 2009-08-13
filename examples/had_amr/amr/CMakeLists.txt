# Copyright (c) 2007-2009 Hartmut Kaiser
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying 
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

###############################################################################
set(amr_client_HEADERS
    amr_mesh.hpp
    functional_component.hpp
    stencil_value.hpp
    dynamic_stencil_value.hpp)
source_group("Header Files" 
    FILES ${amr_client_HEADERS})

set(amr_stubs_HEADERS
    stubs/amr_mesh.hpp
    stubs/functional_component.hpp
    stubs/stencil_value.hpp
    stubs/dynamic_stencil_value.hpp)
source_group("Header Files\\Stubs" 
    FILES ${amr_stubs_HEADERS})

set(amr_server_HEADERS
    server/amr_mesh.hpp
    server/functional_component.hpp
    server/stencil_value.hpp
    server/stencil_value_impl.hpp
    server/stencil_value_in_adaptor.hpp
    server/stencil_value_out_adaptor.hpp
    server/dynamic_stencil_value.hpp
    server/dynamic_stencil_value_impl.hpp)
source_group("Header Files\\Server" 
    FILES ${amr_server_HEADERS})

set(amr_HEADERS
    ${amr_client_HEADERS}
    ${amr_stubs_HEADERS}
    ${amr_server_HEADERS})

###############################################################################
set(amr_client_SOURCES
    amr.cpp)
source_group("Source Files" 
    FILES ${amr_client_SOURCES})

set(amr_server_SOURCES
    server/amr_mesh.cpp
    server/functional_component.cpp
    server/stencil_value.cpp
    server/stencil_value_out_adaptor.cpp
    server/dynamic_stencil_value.cpp)
source_group("Source Files\\Server" 
    FILES ${amr_server_SOURCES})

set(amr_SOURCES
    ${amr_client_SOURCES}
    ${amr_server_SOURCES})

###############################################################################
set(amr_INI
    amr.ini)
source_group("" 
    FILES ${amr_INI})

###############################################################################
add_hpx_component(had_amr 
    SOURCES ${amr_SOURCES} 
    HEADERS ${amr_HEADERS}
    INI ${amr_INI})

