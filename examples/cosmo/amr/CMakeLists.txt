# Copyright (c) 2007-2009 Hartmut Kaiser
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying 
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

###############################################################################
set(cosmo_amr_client_HEADERS
    unigrid_mesh.hpp
    functional_component.hpp
    dynamic_stencil_value.hpp)
source_group("Header Files" 
    FILES ${cosmo_amr_client_HEADERS})

set(cosmo_amr_stubs_HEADERS
    stubs/unigrid_mesh.hpp
    stubs/functional_component.hpp
    stubs/dynamic_stencil_value.hpp)
source_group("Header Files\\Stubs" 
    FILES ${cosmo_amr_stubs_HEADERS})

set(cosmo_amr_server_HEADERS
    server/unigrid_mesh.hpp
    server/functional_component.hpp
    server/stencil_value_in_adaptor.hpp
    server/stencil_value_out_adaptor.hpp
    server/dynamic_stencil_value.hpp
    server/dynamic_stencil_value_impl.hpp)
source_group("Header Files\\Server" 
    FILES ${cosmo_amr_server_HEADERS})

set(cosmo_amr_HEADERS
    ${cosmo_amr_client_HEADERS}
    ${cosmo_amr_stubs_HEADERS}
    ${cosmo_amr_server_HEADERS})

###############################################################################
set(cosmo_amr_client_SOURCES
    amr.cpp parameter.cpp)
source_group("Source Files" 
    FILES ${cosmo_amr_client_SOURCES})

set(cosmo_amr_server_SOURCES
    server/unigrid_mesh.cpp
    server/functional_component.cpp
    server/stencil_value_out_adaptor.cpp
    server/dynamic_stencil_value.cpp)
source_group("Source Files\\Server" 
    FILES ${cosmo_amr_server_SOURCES})

set(cosmo_amr_SOURCES
    ${cosmo_amr_client_SOURCES}
    ${cosmo_amr_server_SOURCES})

###############################################################################
set(cosmo_amr_INI
    cosmo_amr.ini)
source_group("" 
    FILES ${cosmo_amr_INI})

###############################################################################
add_hpx_component(cosmo_amr 
    SOURCES ${cosmo_amr_SOURCES} 
    HEADERS ${cosmo_amr_HEADERS}
    INI ${cosmo_amr_INI})

