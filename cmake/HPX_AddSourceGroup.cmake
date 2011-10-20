# Copyright (c) 2007-2011 Hartmut Kaiser
# Copyright (c) 2011      Bryce Lelbach
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(HPX_ADDSOURCEGROUP_LOADED TRUE)

hpx_include(ParseArguments)

macro(add_hpx_source_group)
  if(MSVC)
    hpx_parse_arguments(GROUP "CLASS;ROOT;TARGETS" "" ${ARGN})

    set(targets "${GROUP_TARGETS}")

    foreach(target ${GROUP_TARGETS})
      string(REGEX REPLACE "${GROUP_ROOT}" "" relpath "${target}")
      string(REGEX REPLACE "[\\\\/][^\\\\/]*$" "" relpath "${relpath}")
      string(REGEX REPLACE "^[\\\\/]" "" relpath "${relpath}")
      string(REGEX REPLACE "/" "\\\\\\\\" relpath "${relpath}")

      if(GROUP_CLASS)
        source_group("${GROUP_CLASS}\\${relpath}" FILES ${target})
      else()
        source_group("${relpath}" FILES ${target})
      endif()
    endforeach()
  endif()
endmacro()

