# Copyright (c) 2011 Bryce Lelbach
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying 
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

# user version
add_hpx_component(agas_stdmap
  MODULE core
  SOURCES component_module.cpp
          component_namespace.cpp
          symbol_namespace.cpp
  ESSENTIAL)

# system version
add_hpx_library(agas_stdmap_bs
  MODULE core
  SOURCES component_module.cpp
          component_namespace.cpp
          symbol_namespace.cpp
  ESSENTIAL)

set_property(TARGET agas_stdmap_bs_lib APPEND
  PROPERTY COMPILE_DEFINITIONS
  "HPX_AGAS_SYSTEM")

