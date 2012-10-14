# Copyright (c) 2011-2012 Bryce Lelbach
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(HPX_APPENDPROPERTY_LOADED TRUE)

include(HPX_Include)

hpx_include(ParseArguments)

macro(hpx_append_property target property value)
  get_target_property(existing ${target} ${property})
  if(existing)
    set(value "${existing} ${value}")
  endif()
  set_target_properties(${target} PROPERTIES ${property} ${value})
endmacro()

