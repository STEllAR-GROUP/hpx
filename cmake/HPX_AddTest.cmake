# Copyright (c) 2011 Bryce Lelbach
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(HPX_ADDTEST_LOADED TRUE)

include(HPX_Include)

hpx_include(ParseArguments)

macro(hpx_make_python_list input output)
  set(${output} "[")
  foreach(element ${${input}})
    set(${output} "${${output}}${element},")
  endforeach()
  set(${output} "${${output}}]")
endmacro()

macro(add_hpx_test category name)
  hpx_parse_arguments(${name} "EXECUTABLE;TIMEOUT;LOCALITIES;THREADS_PER_LOCALITY;ARGS"
                              "FAILURE_EXPECTED" ${ARGN})

  if(NOT ${name}_TIMEOUT)
    set(${name}_TIMEOUT 600)
  endif()

  if(NOT ${name}_LOCALITIES)
    set(${name}_LOCALITIES 1)
  endif()

  if(NOT ${name}_THREADS_PER_LOCALITY)
    set(${name}_THREADS_PER_LOCALITY 1)
  endif()

  if(NOT ${name}_EXECUTABLE)
    set(${name}_EXECUTABLE ${name})
  endif()

  set(expected "True")

  if(${name}_FAILURE_EXPECTED)
    set(expected "False")
  endif()

  set(args)

  foreach(arg ${${name}_ARGS})
    set(args ${args} "'${arg}'")
  endforeach()

  hpx_make_python_list(args ${name}_ARGS)

  set(test_input "'$<TARGET_FILE:${${name}_EXECUTABLE}_test_exe>'"
                 ${${name}_TIMEOUT}
                 ${expected}
                 ${${name}_LOCALITIES}
                 ${${name}_THREADS_PER_LOCALITY}
                 ${${name}_ARGS})

  hpx_make_python_list(test_input test_output)

  set(test_output "  ${test_output},\n")

  add_test(
    NAME "${category}.${name}"
    COMMAND ${CMAKE_SOURCE_DIR}/python/scripts/hpx_run_test.py
            --log-stdout
            "[${test_output}]")
endmacro()

macro(add_hpx_unit_test category name)
  add_hpx_test("tests.unit.${category}" ${name} ${ARGN})
endmacro()

macro(add_hpx_regression_test category name)
  add_hpx_test("tests.regressions.${category}" ${name} ${ARGN})
endmacro()

