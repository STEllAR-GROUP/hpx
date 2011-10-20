# Copyright (c) 2011 Bryce Lelbach
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(HPX_COMPILE_LOADED TRUE)

include(HPX_Include)

hpx_include(ParseArguments)

macro(hpx_compile_object name)
  hpx_parse_arguments(${name}
    "SOURCE;LANGUAGE;FLAGS;OUTPUT" "ESSENTIAL" ${ARGN})

  get_directory_property(definitions DEFINITIONS)

  string(TOUPPER "CMAKE_${${name}_LANGUAGE}_FLAGS_${CMAKE_BUILD_TYPE}"
    build_type_flags)

  set(flags ${${name}_FLAGS}
            ${CMAKE_${${name}_LANGUAGE}_FLAGS}
            ${${build_type_flags}}
            ${definitions})

  # FIXME: This is POSIX only, I think (-c should work on MSVC, not sure about
  # -o, Hartmut says it might be -Fo).
  add_custom_command(OUTPUT ${${name}_OUTPUT}
    COMMAND "${CMAKE_${${name}_LANGUAGE}_COMPILER}" ${flags}
            "-c" "${CMAKE_CURRENT_SOURCE_DIR}/${${name}_SOURCE}"
            "-o" "${${name}_OUTPUT}"
    DEPENDS ${${name}_SOURCE}
    VERBATIM)

  if(${name}_ESSENTIAL)
    add_custom_target(${name} ALL DEPENDS ${${name}_OUTPUT})
  else()
    add_custom_target(${name} DEPENDS ${${name}_OUTPUT})
  endif()
endmacro()

macro(hpx_compile name)
  hpx_parse_arguments(${name}
    "SOURCE;LANGUAGE;FLAGS;OUTPUT" "QUIET" ${ARGN})

  if(${name}_QUIET)
    execute_process(
      COMMAND "${CMAKE_${${name}_LANGUAGE}_COMPILER}" ${${name}_FLAGS}
              "${${name}_SOURCE}"
              "-o" "${${name}_OUTPUT}"
      RESULT_VARIABLE ${name}_RESULT OUTPUT_QUIET ERROR_QUIET)
  else()
    execute_process(
      COMMAND "${CMAKE_${${name}_LANGUAGE}_COMPILER}" ${${name}_FLAGS}
              "${${name}_SOURCE}"
              "-o" "${${name}_OUTPUT}"
      RESULT_VARIABLE ${name}_RESULT
      OUTPUT_FILE ${CMAKE_BINARY_DIR}/${CMAKE_FILES_DIRECTORY}/${name}.stdout
      ERROR_FILE ${CMAKE_BINARY_DIR}/${CMAKE_FILES_DIRECTORY}/${name}.stderr)
  endif()
endmacro()

