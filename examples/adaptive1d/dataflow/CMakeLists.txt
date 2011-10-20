# Copyright (c) 2011      Bryce Lelbach
# Copyright (c) 2007-2009 Hartmut Kaiser
# Copyright (c) 2009-2011 Matthew Anderson
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

###############################################################################
set(client_sources
    dataflow.cpp)

set(server_sources
    server/dataflow_stencil.cpp
    server/functional_component.cpp
    server/stencil_value_out_adaptor.cpp
    server/dynamic_stencil_value.cpp)

###############################################################################
set(client_headers
    dataflow_stencil.hpp
    functional_component.hpp
    dynamic_stencil_value.hpp)

set(stubs_headers
    stubs/dataflow_stencil.hpp
    stubs/functional_component.hpp
    stubs/dynamic_stencil_value.hpp)

set(server_headers
    server/dataflow_stencil.hpp
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
add_hpx_component(adaptive1d_dataflow
  MODULE adaptive1d
  SOURCES ${sources}
  HEADERS ${headers}
  DEPENDENCIES adaptive1d_stencil_component
               distributing_factory_component)

