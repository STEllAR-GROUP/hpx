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
        " ${${name}_FLAGS} ${${name}_SOURCE} "
        " ${outflag} ${${name}_${${name}_LANGUAGE}_COMPILEROUTNAME}")
    execute_process(
      COMMAND "${CMAKE_${${name}_LANGUAGE}_COMPILER}" ${${name}_FLAGS}
              "${${name}_SOURCE}"
              ${outflag} ${${name}_${${name}_LANGUAGE}_COMPILEROUTNAME}
              RESULT_VARIABLE ${name}_RESULT OUTPUT_QUIET ERROR_VARIABLE ${name}_ERROR_OUTPUT)
  else()
    hpx_debug("compile" "${CMAKE_${${name}_LANGUAGE}_COMPILER}"
        " ${${name}_FLAGS} ${${name}_SOURCE}"
        " ${outflag} ${${name}_${${name}_LANGUAGE}_COMPILEROUTNAME}")
    execute_process(
      COMMAND "${CMAKE_${${name}_LANGUAGE}_COMPILER}" ${${name}_FLAGS}
              "${${name}_SOURCE}"
              ${outflag} ${${name}_${${name}_LANGUAGE}_COMPILEROUTNAME}
      RESULT_VARIABLE ${name}_RESULT
      ERROR_VARIABLE ${name}_STDERR
      OUTPUT_VARIABLE ${name}_STDOUT
      OUTPUT_STRIP_TRAILING_WHITESPACE
      ERROR_STRIP_TRAILING_WHITESPACE
      )
      if(${name}_STDERR)
        if("${CMAKE_${${name}_LANGUAGE}_COMPILER_ID}" STREQUAL "Intel")
          string(REGEX MATCH ".*command line warning #.*" ${name}_HAS_COMMAND_LINE_WARNING ${${name}_STDERR})
          if(${name}_HAS_COMMAND_LINE_WARNING)
            set(${name}_RESULT "1")
          endif()
        elseif("${CMAKE_${${name}_LANGUAGE}_COMPILER_ID}" STREQUAL "Clang")
          string(REGEX MATCH ".*(argument unused during compilation|unknown warning option).*" ${name}_HAS_UNUSED_ARGUMENT_WARNING ${${name}_STDERR})
        endif()
        file(WRITE ${CMAKE_BINARY_DIR}/${CMAKE_FILES_DIRECTORY}/${name}.${${name}_LANGUAGE}.stderr ${${name}_STDERR})
      endif()
      if(${name}_STDOUT)
        file(WRITE ${CMAKE_BINARY_DIR}/${CMAKE_FILES_DIRECTORY}/${name}.${${name}_LANGUAGE}.stdout ${${name}_STDOUT})
      endif()
  endif()
endmacro()

