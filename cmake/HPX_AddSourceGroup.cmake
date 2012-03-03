# Copyright (c) 2007-2012 Hartmut Kaiser
# Copyright (c) 2011      Bryce Lelbach
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(HPX_ADDSOURCEGROUP_LOADED TRUE)

hpx_include(ParseArguments)

macro(add_hpx_source_group)
  if(MSVC)
    hpx_parse_arguments(GROUP "NAME;CLASS;ROOT;TARGETS" "" ${ARGN})

    set(targets "${GROUP_TARGETS}")

    set(name "")
    if(GROUP_NAME)
      set(name ${GROUP_NAME})
    endif()

    if (NOT GROUP_ROOT)
      set(GROUP_ROOT ".")
    endif()
    get_filename_component(root ${GROUP_ROOT} ABSOLUTE)

    foreach(target ${targets})
      string(REGEX REPLACE "${root}" "" relpath "${target}")
      string(REGEX REPLACE "[\\\\/][^\\\\/]*$" "" relpath "${relpath}")
      string(REGEX REPLACE "^[\\\\/]" "" relpath "${relpath}")
      string(REGEX REPLACE "/" "\\\\\\\\" relpath "${relpath}")

      if(GROUP_CLASS)
        if(NOT ("${relpath}" STREQUAL ""))
          hpx_debug("add_source_group.${name}"
                    "Adding ${target} to source group '${GROUP_CLASS}\\\\${relpath}' (${root}\\\\${relpath})")
          source_group("${GROUP_CLASS}\\${relpath}" FILES ${target})
        endif()
      else()
        hpx_debug("add_source_group.${name}"
                  "Adding ${target} to source group ${root}\\\\${relpath}")
        source_group("${relpath}" FILES ${target})
      endif()
    endforeach()
  endif()
endmacro()

