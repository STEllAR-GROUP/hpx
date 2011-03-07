# Copyright (c) 2011 Bryce Lelbach
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying 
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(HPX_COMPILEOBJECT_LOADED TRUE)

include(HPX_Include)

hpx_include(ParseArguments)

macro(hpx_compile_object name)
  hpx_parse_arguments(${name}
    "MODULE;SOURCE;HEADERS;LANGUAGE;FLAGS;OUTPUT" "ESSENTIAL" ${ARGN})

  get_directory_property(definitions DEFINITIONS)

  string(TOUPPER "CMAKE_${${name}_LANGUAGE}_FLAGS_${CMAKE_BUILD_TYPE}"
    build_type_flags)

  set(flags ${${name}_FLAGS}
            ${CMAKE_C_FLAGS}
            ${${build_type_flags}}
            ${definitions})

  # FIXME: This is POSIX only, I think (-c should work on MSVC, not sure about
  # -o, Hartmut says it might be -Fo).
  add_custom_command(OUTPUT ${${name}_OUTPUT} 
    COMMAND "${CMAKE_C_COMPILER}" ${flags}
            "-c" "${CMAKE_CURRENT_SOURCE_DIR}/${${name}_SOURCE}"
            "-o" "${${name}_OUTPUT}"
    DEPENDS ${${name}_SOURCE} ${${name}_HEADERS}
    VERBATIM)

  if(${name}_ESSENTIAL)
    add_custom_target(${name} ALL DEPENDS ${${name}_OUTPUT})
  else()
    add_custom_target(${name} DEPENDS ${${name}_OUTPUT})
  endif()
endmacro()

