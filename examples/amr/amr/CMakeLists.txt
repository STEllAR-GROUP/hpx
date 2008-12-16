# Copyright (c) 2007-2008 Hartmut Kaiser
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying 
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(amr_HEADERS
    functional_component.hpp
    stencil_value.hpp
    stubs/functional_component.hpp
    stubs/stencil_value.hpp
    server/functional_component.hpp
#    server/functional_timestep.hpp
#    server/functional_timestep_impl.hpp
    server/stencil_value.hpp
    server/stencil_value_impl.hpp
    server/stencil_value_in_adaptor.hpp
    server/stencil_value_out_adaptor.hpp)

set(amr_SOURCES
    amr.cpp
    server/functional_component.cpp
#    server/functional_timestep.cpp
    server/stencil_value.cpp
    server/stencil_value_out_adaptor.cpp)

set(amr_INI
    amr.ini)

add_hpx_component(amr 
    SOURCES ${amr_SOURCES} 
    HEADERS ${amr_HEADERS}
    INI ${amr_INI})

