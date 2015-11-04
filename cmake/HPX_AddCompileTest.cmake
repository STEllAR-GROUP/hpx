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

  add_hpx_library(
    ${name}
    SOURCE_ROOT ${${name}_SOURCE_ROOT}
    SOURCES ${${name}_SOURCES}
    EXCLUDE_FROM_ALL
    EXCLUDE_FROM_DEFAULT_BUILD
    FOLDER ${${name}_FOLDER}
    STATIC)

  add_test(NAME "${category}.${name}"
    COMMAND ${CMAKE_COMMAND}
      --build .
      --target "${name}_lib"
      --config $<CONFIGURATION>
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR})

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

macro(add_hpx_headers_compile_test category name)
  add_hpx_compile_test("tests.headers.${category}" ${name} ${ARGN})
endmacro()

