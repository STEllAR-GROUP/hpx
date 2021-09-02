# Copyright (c) 2007-2012 Hartmut Kaiser
# Copyright (c) 2011      Bryce Lelbach
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

function(hpx_local_add_library_headers name globtype)
  if(MSVC)
    set(options APPEND)
    set(one_value_args)
    set(multi_value_args EXCLUDE GLOBS)
    cmake_parse_arguments(
      HEADERS "${options}" "${one_value_args}" "${multi_value_args}" ${ARGN}
    )

    if(NOT HEADERS_APPEND)
      set(${name}_HEADERS
          ""
          CACHE INTERNAL "Headers for lib${name}." FORCE
      )
    endif()

    file(${globtype} headers ${HEADERS_GLOBS})

    foreach(header ${headers})
      get_filename_component(absolute_path ${header} ABSOLUTE)

      set(add_flag ON)

      if(HEADERS_EXCLUDE)
        if(${absolute_path} MATCHES ${HEADERS_EXCLUDE})
          set(add_flag OFF)
        endif()
      endif()

      if(add_flag)
        hpx_local_debug(
          "add_library_headers.${name}"
          "Adding ${absolute_path} to header list for lib${name}"
        )
        set(${name}_HEADERS
            ${${name}_HEADERS} ${absolute_path}
            CACHE INTERNAL "Headers for lib${name}." FORCE
        )
      endif()
    endforeach()
  endif()
endfunction()

# ##############################################################################
function(hpx_local_add_library_headers_noglob name)
  if(MSVC)
    set(options APPEND)
    set(one_value_args)
    set(multi_value_args EXCLUDE HEADERS)
    cmake_parse_arguments(
      HEADERS "${options}" "${one_value_args}" "${multi_value_args}" ${ARGN}
    )

    hpx_local_print_list(
      "DEBUG" "hpx_local_add_library_sources_noglob.${name}"
      "Sources for ${name}" HEADERS_HEADERS
    )

    set(headers ${HEADERS_HEADERS})

    if(NOT HEADERS_APPEND)
      set(${name}_HEADERS
          ""
          CACHE INTERNAL "Headers for lib${name}." FORCE
      )
    endif()

    foreach(header ${headers})
      get_filename_component(absolute_path ${header} ABSOLUTE)

      set(add_flag ON)

      if(HEADERS_EXCLUDE)
        if(${absolute_path} MATCHES ${HEADERS_EXCLUDE})
          set(add_flag OFF)
        endif()
      endif()

      if(add_flag)
        hpx_local_debug(
          "hpx_local_add_library_headers_noglob.${name}"
          "Adding ${absolute_path} to header list for lib${name}"
        )
        set(${name}_HEADERS
            ${${name}_HEADERS} ${absolute_path}
            CACHE INTERNAL "Headers for lib${name}." FORCE
        )
      endif()
    endforeach()
  endif()
endfunction()
