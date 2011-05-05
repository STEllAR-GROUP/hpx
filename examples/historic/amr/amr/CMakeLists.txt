# Copyright (c) 2007-2009 Hartmut Kaiser
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying 
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

###############################################################################
set(client_sources
    amr.cpp)

set(server_sources
    server/amr_mesh.cpp
    server/functional_component.cpp
    server/stencil_value.cpp
    server/stencil_value_out_adaptor.cpp)

###############################################################################
set(client_headers
    amr_mesh.hpp
    functional_component.hpp
    stencil_value.hpp)

set(stubs_headers
    stubs/amr_mesh.hpp
    stubs/functional_component.hpp
    stubs/stencil_value.hpp)

set(server_headers
    server/amr_mesh.hpp
    server/functional_component.hpp
    server/stencil_value.hpp
    server/stencil_value_impl.hpp
    server/stencil_value_in_adaptor.hpp
    server/stencil_value_out_adaptor.hpp)

###############################################################################
set(sources
    ${client_sources}
    ${server_sources})

set(headers
    ${client_headers}
    ${stubs_headers}
    ${server_headers})

###############################################################################
set(ini_files
    amr.ini)

###############################################################################
source_group("Source Files\\Client" FILES ${client_sources})
source_group("Source Files\\Server" FILES ${server_sources})
source_group("Header Files\\Client" FILES ${client_headers})
source_group("Header Files\\Stubs"  FILES ${stubs_headers})
source_group("Header Files\\Server" FILES ${server_headers})
source_group("Configuration Files"  FILES ${ini_files})

###############################################################################
add_hpx_component(amr 
  MODULE amr
  SOURCES ${sources} 
  HEADERS ${headers}
  INI ${ini_files})

