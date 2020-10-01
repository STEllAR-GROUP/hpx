# Copyright (c) 2007-2012 Hartmut Kaiser
# Copyright (c) 2011      Bryce Lelbach
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

function(add_hpx_source_group)
  if(MSVC)
    set(options)
    set(one_value_args NAME CLASS ROOT)
    set(multi_value_args TARGETS)
    cmake_parse_arguments(
      GROUP "${options}" "${one_value_args}" "${multi_value_args}" ${ARGN}
    )

    set(name "")
    if(GROUP_NAME)
      set(name ${GROUP_NAME})
    endif()

    if(NOT GROUP_ROOT)
      set(GROUP_ROOT ".")
    endif()
    get_filename_component(root "${GROUP_ROOT}" ABSOLUTE)

    hpx_debug("add_source_group.${name}" "root: ${GROUP_ROOT}")

    foreach(target ${GROUP_TARGETS})
      string(REGEX REPLACE "${root}" "" relpath "${target}")
      set(_target ${relpath})
      string(REGEX REPLACE "[\\\\/][^\\\\/]*$" "" relpath "${relpath}")
      string(REGEX REPLACE "^[\\\\/]" "" relpath "${relpath}")
      string(REGEX REPLACE "/" "\\\\\\\\" relpath "${relpath}")

      if(GROUP_CLASS)
        hpx_debug(
          "add_source_group.${name}"
          "Adding '${target}' to source group '${GROUP_CLASS}', sub-group '${relpath}'"
        )
        source_group("${GROUP_CLASS}\\${relpath}" FILES ${target})
      else()
        hpx_debug("add_source_group.${name}"
                  "Adding ${target} to source group ${relpath}"
        )
        source_group("${relpath}" FILES ${target})
      endif()
    endforeach()
  endif()
endfunction()
