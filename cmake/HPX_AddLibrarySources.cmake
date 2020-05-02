# Copyright (c) 2007-2012 Hartmut Kaiser
# Copyright (c) 2011      Bryce Lelbach
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

function(add_hpx_library_sources name globtype)
  set(options APPEND)
  set(one_value_args)
  set(multi_value_args EXCLUDE GLOBS)
  cmake_parse_arguments(
    SOURCES "${options}" "${one_value_args}" "${multi_value_args}" ${ARGN}
  )

  file(${globtype} sources ${SOURCES_GLOBS})

  if(NOT SOURCES_APPEND)
    set(${name}_SOURCES
        ""
        CACHE INTERNAL "Sources for lib${name}." FORCE
    )
  endif()

  foreach(source ${sources})
    get_filename_component(absolute_path ${source} ABSOLUTE)

    set(add_flag ON)

    if(SOURCES_EXCLUDE)
      if(${absolute_path} MATCHES ${SOURCES_EXCLUDE})
        set(add_flag OFF)
      endif()
    endif()

    if(add_flag)
      hpx_debug("add_library_sources.${name}"
                "Adding ${absolute_path} to source list for lib${name}"
      )
      set(${name}_SOURCES
          ${${name}_SOURCES} ${absolute_path}
          CACHE INTERNAL "Sources for lib${name}." FORCE
      )
    endif()
  endforeach()
endfunction()

# ##############################################################################
function(add_hpx_library_sources_noglob name)
  set(options APPEND)
  set(one_value_args)
  set(multi_value_args EXCLUDE SOURCES)
  cmake_parse_arguments(
    SOURCES "${options}" "${one_value_args}" "${multi_value_args}" ${ARGN}
  )

  # hpx_print_list("DEBUG" "add_hpx_library_sources_noglob.${name}" "Sources for
  # ${name}" ${SOURCES_SOURCES})

  set(sources ${SOURCES_SOURCES})

  if(NOT SOURCES_APPEND)
    set(${name}_SOURCES
        ""
        CACHE INTERNAL "Sources for lib${name}." FORCE
    )
  endif()

  foreach(source ${sources})
    get_filename_component(absolute_path ${source} ABSOLUTE)

    set(add_flag ON)

    if(SOURCES_EXCLUDE)
      if(${absolute_path} MATCHES ${SOURCES_EXCLUDE})
        set(add_flag OFF)
      endif()
    endif()

    if(add_flag)
      hpx_debug("add_library_sources.${name}"
                "Adding ${absolute_path} to source list for lib${name}"
      )
      set(${name}_SOURCES
          ${${name}_SOURCES} ${absolute_path}
          CACHE INTERNAL "Sources for lib${name}." FORCE
      )
    endif()
  endforeach()
endfunction()
