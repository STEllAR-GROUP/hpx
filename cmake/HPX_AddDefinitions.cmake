# Copyright (c) 2013 Hartmut Kaiser
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(HPX_ADD_DEFINITION_LOADED TRUE)

include(HPX_Include)

hpx_include(Message)

set(HPX_PREPROCESS_DEFINITIONS CACHE INTERNAL "" FORCE)

macro(hpx_add_definitions definition)
  hpx_debug("add_definitions" "${definition}")
  if(HPX_AUTOMATIC_PREPROCESSING AND ${BOOST_MINOR_VERSION} GREATER 50)
    set(HPX_PREPROCESS_DEFINITIONS ${HPX_PREPROCESS_DEFINITIONS} ${definition})
  endif()
  add_definitions(${definition})
endmacro()

