# Copyright (c) 2011 Bryce Lelbach
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying 
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

if(NOT MSVC)
  add_hpx_component(agas_stdmap_core 
    MODULE core
    SOURCES component_module.cpp
            component_namespace.cpp
            symbol_namespace.cpp
    ESSENTIAL)
endif()

