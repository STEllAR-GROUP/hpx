# Copyright (c) 2011 Bryce Lelbach
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

macro(add_hpx_test category name)
  set(options FAILURE_EXPECTED)
  set(one_value_args EXECUTABLE LOCALITIES THREADS_PER_LOCALITY)
  set(multi_value_args ARGS)
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
  else()
    set(_exe "${${name}_EXECUTABLE}")
  endif()

  set(expected "0")

  if(${name}_FAILURE_EXPECTED)
    set(expected "1")
  endif()

  set(args)

  foreach(arg ${${name}_ARGS})
    set(args ${args} "'${arg}'")
  endforeach()
  set(args ${args} "-v" "--" ${args})

  set(cmd "${PYTHON_EXECUTABLE}"
          "${CMAKE_BINARY_DIR}/bin/hpxrun.py"
          ${_exe}
          "-e" "${expected}"
          "-l" "${${name}_LOCALITIES}"
          "-t" "${${name}_THREADS_PER_LOCALITY}")

  if(${name}_LOCALITIES STREQUAL "1")
    add_test(
      NAME "${category}.${name}"
      COMMAND ${cmd} ${args})
    else()
      if(HPX_PARCELPORT_IBVERBS)
        add_test(
          NAME "${category}.distributed.ibverbs.${name}"
          COMMAND ${cmd} "-p" "ibverbs" ${args})
      endif()
      if(HPX_PARCELPORT_IPC)
        add_test(
          NAME "${category}.distributed.ipc.${name}"
          COMMAND ${cmd} "-p" "ipc" ${args})
      endif()
      if(HPX_PARCELPORT_MPI)
        add_test(
          NAME "${category}.distributed.mpi.${name}"
          COMMAND ${cmd} "-p" "mpi" "-r" "mpi" ${args})
      endif()
      if(HPX_PARCELPORT_TCP)
        add_test(
          NAME "${category}.distributed.tcp.${name}"
          COMMAND ${cmd} "-p" "tcp" ${args})
      endif()
    endif()
endmacro()

macro(add_hpx_unit_test category name)
  add_hpx_test("tests.unit.${category}" ${name} ${ARGN})
endmacro()

macro(add_hpx_regression_test category name)
  add_hpx_test("tests.regressions.${category}" ${name} ${ARGN})
endmacro()

