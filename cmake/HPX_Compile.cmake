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

  if(NOT MSVC)
    set(outflag "-o")
    set(${name}_${${name}_LANGUAGE}_COMPILEROUTNAME "${${name}_OUTPUT}")
  else()
    set(outflag "")
    set(${name}_${${name}_LANGUAGE}_COMPILEROUTNAME "-Fo${${name}_OUTPUT}")
  endif()

  add_custom_command(OUTPUT ${${name}_OUTPUT}
    COMMAND "${CMAKE_${${name}_LANGUAGE}_COMPILER}" ${flags}
            "-c" "${CMAKE_CURRENT_SOURCE_DIR}/${${name}_SOURCE}"
            ${outflag} ${${name}_${${name}_LANGUAGE}_COMPILEROUTNAME}
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

  if(NOT MSVC)
    set(outflag "-o")
    set(${name}_${${name}_LANGUAGE}_COMPILEROUTNAME "${${name}_OUTPUT}")
  else()
    set(outflag "")
    set(${name}_${${name}_LANGUAGE}_COMPILEROUTNAME "-Fo${${name}_OUTPUT}")
  endif()

  if(${name}_QUIET)
    hpx_debug("compile" "${CMAKE_${${name}_LANGUAGE}_COMPILER}"
        "${${name}_FLAGS} ${${name}_SOURCE}"
        ${outflag} "${${name}_${${name}_LANGUAGE}_COMPILEROUTNAME}")
    execute_process(
      COMMAND "${CMAKE_${${name}_LANGUAGE}_COMPILER}" ${${name}_FLAGS}
              "${${name}_SOURCE}"
              ${outflag} ${${name}_${${name}_LANGUAGE}_COMPILEROUTNAME}
      RESULT_VARIABLE ${name}_RESULT OUTPUT_QUIET ERROR_QUIET)
  else()
    hpx_debug("compile" "${CMAKE_${${name}_LANGUAGE}_COMPILER}"
        "${${name}_FLAGS} ${${name}_SOURCE}"
        ${outflag} "${${name}_${${name}_LANGUAGE}_COMPILEROUTNAME}")
    execute_process(
      COMMAND "${CMAKE_${${name}_LANGUAGE}_COMPILER}" ${${name}_FLAGS}
              "${${name}_SOURCE}"
              ${outflag} ${${name}_${${name}_LANGUAGE}_COMPILEROUTNAME}
      RESULT_VARIABLE ${name}_RESULT
      OUTPUT_FILE ${CMAKE_BINARY_DIR}/${CMAKE_FILES_DIRECTORY}/${name}.stdout
      ERROR_FILE ${CMAKE_BINARY_DIR}/${CMAKE_FILES_DIRECTORY}/${name}.stderr)
  endif()
endmacro()

