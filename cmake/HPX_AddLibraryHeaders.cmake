# Copyright (c) 2007-2012 Hartmut Kaiser
# Copyright (c) 2011      Bryce Lelbach
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(HPX_ADDLIBRARYHEADERS_LOADED TRUE)

hpx_include(Message
            ParseArguments)

macro(add_hpx_library_headers name globtype)
  if(MSVC)
    hpx_parse_arguments(HEADERS "EXCLUDE;GLOBS" "APPEND" ${ARGN})

    if(NOT ${HEADERS_APPEND})
      set(${name}_HEADERS "" CACHE INTERNAL "Headers for lib${name}." FORCE)
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
        hpx_debug("add_library_headers.${name}"
                  "Adding ${absolute_path} to header list for lib${name}")
        set(${name}_HEADERS ${${name}_HEADERS} ${absolute_path}
          CACHE INTERNAL "Headers for lib${name}." FORCE)
      endif()
    endforeach()
  endif()
endmacro()

###############################################################################
macro(add_hpx_library_headers_noglob name)
  if(MSVC)
    hpx_parse_arguments(HEADERS "EXCLUDE;HEADERS" "APPEND" ${ARGN})

#    hpx_print_list("DEBUG" "add_hpx_library_sources_noglob.${name}"
#      "Sources for ${name}" ${HEADERS_HEADERS})

    set(headers ${HEADERS_HEADERS})

    if(NOT ${HEADERS_APPEND})
      set(${name}_HEADERS "" CACHE INTERNAL "Headers for lib${name}." FORCE)
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
        hpx_debug("add_library_headers.${name}"
                  "Adding ${absolute_path} to header list for lib${name}")
        set(${name}_HEADERS ${${name}_HEADERS} ${absolute_path}
          CACHE INTERNAL "Headers for lib${name}." FORCE)
      endif()
    endforeach()
  endif()
endmacro()

