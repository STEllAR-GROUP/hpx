# Copyright (c) 2013 Hartmut Kaiser
# Copyright (c) 2014 Thomas Heller
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

if(NOT DEFINED HPX_CONFIG_DEFINITIONS)
  set(HPX_CONFIG_DEFINITIONS "" CACHE INTERNAL "" FORCE)
endif()

macro(hpx_add_config_define definition)
  if(NOT DEFINED ${definition}_define)
    set(${definition}_define ${ARGN} CACHE INTERNAL "${definition}" FORCE)
    set(HPX_CONFIG_DEFINITIONS ${HPX_CONFIG_DEFINITIONS} ${definition} CACHE INTERNAL "" FORCE)
  endif()
endmacro()
