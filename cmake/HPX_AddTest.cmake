# Copyright (c) 2011 Bryce Lelbach
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

macro(add_hpx_test category name)
  set(options FAILURE_EXPECTED)
  set(one_value_args EXECUTABLE LOCALITIES THREADS_PER_LOCALITY)
  set(multi_value_args ARGS PARCELPORTS)
  cmake_parse_arguments(${name} "${options}" "${one_value_args}" "${multi_value_args}" ${ARGN})

  if(NOT ${name}_LOCALITIES)
    set(${name}_LOCALITIES 1)
  endif()

  if(NOT ${name}_THREADS_PER_LOCALITY)
    set(${name}_THREADS_PER_LOCALITY 1)
  endif()

  if(NOT ${name}_EXECUTABLE)
    set(${name}_EXECUTABLE ${name})
  endif()

  if(TARGET ${${name}_EXECUTABLE}_test_exe)
    set(_exe "$<TARGET_FILE:${${name}_EXECUTABLE}_test_exe>")
  elseif(TARGET ${${name}_EXECUTABLE}_exe)
    set(_exe "$<TARGET_FILE:${${name}_EXECUTABLE}_exe>")
  else()
    set(_exe "${${name}_EXECUTABLE}")
  endif()

  set(args)

  foreach(arg ${${name}_UNPARSED_ARGUMENTS})
    set(args ${args} "${arg}")
  endforeach()

  if(${name}_LOCALITIES EQUAL "1")
    set(_cmd ${_exe} --hpx:threads ${${name}_THREADS_PER_LOCALITY})
    add_test(
      NAME "${category}.${name}"
      COMMAND ${_cmd} ${args})
    set_tests_properties("${category}.${name}" PROPERTIES WILL_FAIL ${name}_FAILURE_EXPECTED)
  else()
    message("Test ${name} is a multi-locality test " ${${name}_LOCALITIES})
    if(NOT HPX_WITH_NETWORKING)
      message(FATAL_ERROR "Adding a multi-locality test with networking disabled")
    endif()
    # add distributed versions of each multi-locality test
    foreach(_pp ${HPX_STATIC_PARCELPORT_PLUGINS}) # ${${name}_PARCELPORTS})
      message("Adding test for parcelport ${_pp}")
      if (${_pp} STREQUAL "mpi")
        set(_cmd
          "${MPIEXEC_EXECUTABLE}" "${MPIEXEC_PREFLAGS}" "${MPIEXEC_NUMPROC_FLAG}" "${${name}_LOCALITIES}"
          "${_exe}"
          --hpx:threads ${${name}_THREADS_PER_LOCALITY}
        )
      else()
        set(expected "0")
        if(${name}_FAILURE_EXPECTED)
          set(expected "1")
        endif()
        set(_cmd "${PYTHON_EXECUTABLE}"
          "${CMAKE_BINARY_DIR}/bin/hpxrun.py"
          ${_exe}
          "-e" "${expected}"
          "-t" "${${name}_THREADS_PER_LOCALITY}"
          "-l" "${${name}_LOCALITIES}"
          "-p" "${_pp}"
          "-v" -- ${args})
      endif()
      add_test(
        NAME "${category}.distributed.${_pp}.${name}"
        COMMAND ${_cmd}
      )
      set_tests_properties("${category}.distributed.${_pp}.${name}" PROPERTIES WILL_FAIL ${name}_FAILURE_EXPECTED)
    endforeach()
  endif()
endmacro()

macro(add_hpx_unit_test category name)
  add_hpx_test("tests.unit.${category}" ${name} ${ARGN})
endmacro()

macro(add_hpx_regression_test category name)
  add_hpx_test("tests.regressions.${category}" ${name} ${ARGN})
endmacro()

macro(add_hpx_example_test category name)
  add_hpx_test("tests.examples.${category}" ${name} ${ARGN})
endmacro()


