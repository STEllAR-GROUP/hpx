# Copyright (c) 2013 Hartmut Kaiser
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(HPX_ADD_DEFINITION_LOADED TRUE)

include(HPX_Include)

hpx_include(Message
            ParseArguments)

macro(hpx_add_definitions definition)
  hpx_debug("add_definitions" "${definition}")
  set(HPX_DEFINITIONS ${HPX_DEFINITIONS} ${definition})
  add_definitions(${definition})
endmacro()

macro(hpx_add_config_define definition)
  set(${definition}_define ${ARGN} CACHE INTERNAL "${definition}" FORCE)
  set(HPX_CONFIG_DEFINITIONS ${HPX_CONFIG_DEFINITIONS} ${definition})
endmacro()
