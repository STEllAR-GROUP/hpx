# Copyright (c) 2007-2009 Hartmut Kaiser
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying 
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

###############################################################################
set(sources
    generic_module.cpp
    generic_component_noresult.cpp
    generic_component_result.cpp)

set(headers
    generic_component_noresult.hpp
    generic_component_result.hpp)

###############################################################################
source_group("Source Files" FILES ${sources})
source_group("Header Files" FILES ${headers})

###############################################################################
add_hpx_component(generic
  MODULE generic
  SOURCES ${sources} 
  HEADERS ${headers})

