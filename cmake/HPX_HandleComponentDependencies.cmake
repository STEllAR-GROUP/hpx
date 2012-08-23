# Copyright (c) 2007-2012 Hartmut Kaiser
# Copyright (c) 2011      Bryce Lelbach
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(HPX_HANDLECOMPONENTDEPENDENCIES_LOADED TRUE)

include(HPX_Include)

hpx_include(IsTarget)

macro(hpx_handle_component_dependencies components)
  if(NOT MSVC)
    set(prefix "hpx_component_")
  else()
    set(prefix "")
  endif()

  set(tmp "")

  foreach(component ${${components}})
    hpx_is_target(is_target ${component})
    if(is_target)
      set(tmp ${tmp} ${component}_component) 
    else()
      if(HPX_EXTERNAL_CMAKE AND "${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
        set(tmp ${tmp} ${prefix}${component}${HPX_DEBUG_POSTFIX})
      else()
        set(tmp ${tmp} ${prefix}${component})
      endif()
    endif()
  endforeach()

  set(${components} ${tmp})
endmacro()

