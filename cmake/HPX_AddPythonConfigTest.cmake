# Copyright (c) 2011 Bryce Lelbach
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(HPX_ADDPYTHONCONFIGTEST_LOADED TRUE)

include(HPX_Include)

hpx_include(Message
            ParseArguments)

macro(add_hpx_python_config_test name variable)
  hpx_parse_arguments(${name} "SOURCE;ARGS;DEFINITIONS" "" ${ARGN})

  if("${variable}" STREQUAL "ON")
    set(${variable} ON CACHE STRING "${name} state.")
    foreach(definition ${${name}_DEFINITIONS})
      hpx_add_config_define(${definition})
    endforeach()
  elseif("${variable}" STREQUAL "OFF")
    set(${variable} OFF CACHE STRING "${name} state.")
  else()
    file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/${CMAKE_FILES_DIRECTORY}/config_tests)

    set(test_source "")

    if(${name}_ROOT)
      set(test_source "${${name}_ROOT}/share/hpx/${${name}_SOURCE}")
    else()
      set(test_source "${hpx_SOURCE_DIR}/${${name}_SOURCE}")
    endif()

    hpx_debug("python_config_test.${name}" "Using ${test_source} as source file.")

    set(test_result 0)

    execute_process(
      COMMAND "${PYTHON_EXECUTABLE}" "${test_source}" ${${name}_ARGS}
      RESULT_VARIABLE test_result OUTPUT_QUIET ERROR_QUIET)

    if("${test_result}" STREQUAL "0")
      set(${variable} ON CACHE STRING "${name} state.")
      foreach(definition ${${name}_DEFINITIONS})
        hpx_add_config_define(${definition})
      endforeach()
      hpx_info("python_config_test.${name}" "Test passed.")
    else()
      set(${variable} OFF CACHE STRING "${name} state.")
      hpx_warn("python_config_test.${name}" "Test failed, returned ${test_result}.")
    endif()
  endif()
endmacro()

###############################################################################
macro(hpx_check_for_python_paramiko variable)
  add_hpx_python_config_test("python_paramiko" ${variable}
    SOURCE cmake/tests/python_paramiko.py ${ARGN})
endmacro()

macro(hpx_check_for_python_optparse variable)
  add_hpx_python_config_test("python_optparse" ${variable}
    SOURCE cmake/tests/python_optparse.py ${ARGN})
endmacro()

macro(hpx_check_for_python_threading variable)
  add_hpx_python_config_test("python_threading" ${variable}
    SOURCE cmake/tests/python_threading.py ${ARGN})
endmacro()

macro(hpx_check_for_python_with_statement variable)
  add_hpx_python_config_test("python_with_statement" ${variable}
    SOURCE cmake/tests/python_with_statement.py ${ARGN})
endmacro()

macro(hpx_check_for_python_subprocess variable)
  add_hpx_python_config_test("python_subprocess" ${variable}
    SOURCE cmake/tests/python_subprocess.py ${ARGN})
endmacro()

