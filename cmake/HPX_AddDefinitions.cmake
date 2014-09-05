# Copyright (c) 2013 Hartmut Kaiser
# Copyright (c) 2014 Thomas Heller
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

# on startup, this is unset, but we'll set it to an empty string anyway
set_property(GLOBAL PROPERTY HPX_CONFIG_DEFINITIONS "")

function(hpx_add_config_define definition)
 
  if(ARGN)
    set_property(GLOBAL APPEND PROPERTY HPX_CONFIG_DEFINITIONS "${definition} ${ARGN}")
  else()
    set_property(GLOBAL APPEND PROPERTY HPX_CONFIG_DEFINITIONS "${definition}")
  endif()

  get_property(HPX_CONFIG_DEFINITIONS_VAR GLOBAL PROPERTY HPX_CONFIG_DEFINITIONS)

endfunction()
