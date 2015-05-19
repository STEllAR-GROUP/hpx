# Copyright (c) 2015 Thomas Heller
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

macro(add_hpx_compile_test category name)
  set(options FAILURE_EXPECTED)
  set(one_value_args SOURCE_ROOT FOLDER)
  set(multi_value_args SOURCES)

  cmake_parse_arguments(${name} "${options}" "${one_value_args}" "${multi_value_args}" ${ARGN})

  set(expected FALSE)

  if(${name}_FAILURE_EXPECTED)
    set(expected TRUE)
  endif()

  get_directory_property(_INCLUDE_DIRS INCLUDE_DIRECTORIES)
  foreach(dir ${_INCLUDE_DIRS})
    if(NOT MSVC)
      set(include_flags ${include_flags} "-I${dir}")
    else()
      set(include_flags ${include_flags} "/I ${dir}")
    endif()
  endforeach()

  if(NOT ${name}_SOURCE_ROOT)
    set(${name}_SOURCE_ROOT ".")
  endif()
  add_hpx_source_group(
    NAME ${name}
    CLASS "Source Files"
    ROOT ${${name}_SOURCE_ROOT}
    TARGETS ${${name}_SOURCES})

  set(sources)
  foreach(source ${${name}_SOURCES})
    set(sources ${sources} "${CMAKE_CURRENT_SOURCE_DIR}/${source}")
  endforeach()

  string(REPLACE " " ";" CMAKE_CXX_FLAGS_LIST ${CMAKE_CXX_FLAGS})
  set(cmd
    ${CMAKE_CXX_COMPILER} ${CMAKE_CXX_FLAGS_LIST} ${include_flags} ${sources}
  )

  if(MSVC)
    set(cmd ${cmd} -c /Fo"${CMAKE_FILES_DIRECTORY}")
  else()
    set(cmd ${cmd} -c -o "${CMAKE_FILES_DIRECTORY}/${name}.o")
  endif()

  add_test(
    NAME "${category}.${name}"
    COMMAND ${cmd}
  )
  if(expected)
    set_tests_properties("${category}.${name}" PROPERTIES WILL_FAIL TRUE)
  endif()

endmacro()

macro(add_hpx_unit_compile_test category name)
  add_hpx_compile_test("tests.unit.${category}" ${name} ${ARGN})
endmacro()

macro(add_hpx_regression_compile_test category name)
  add_hpx_compile_test("tests.regressions.${category}" ${name} ${ARGN})
endmacro()

