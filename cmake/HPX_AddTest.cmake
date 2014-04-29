# Copyright (c) 2011 Bryce Lelbach
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(HPX_ADDTEST_LOADED TRUE)

include(HPX_Include)

hpx_include(ParseArguments)

macro(add_hpx_test category name)
  hpx_parse_arguments(${name} "EXECUTABLE;LOCALITIES;THREADS_PER_LOCALITY;ARGS"
                              "FAILURE_EXPECTED" ${ARGN})

  if(NOT ${name}_LOCALITIES)
    set(${name}_LOCALITIES 1)
  endif()

  if(NOT ${name}_THREADS_PER_LOCALITY)
    set(${name}_THREADS_PER_LOCALITY 1)
  endif()

  if(NOT ${name}_EXECUTABLE)
    set(${name}_EXECUTABLE ${name})
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
          "$<TARGET_FILE:${${name}_EXECUTABLE}_test_exe>"
          "-e" "${expected}"
          "-l" "${${name}_LOCALITIES}"
          "-t" "${${name}_THREADS_PER_LOCALITY}")

  if("${${name}_LOCALITIES}" STREQUAL "1")
    add_test(
      NAME "${category}.${name}"
      COMMAND ${cmd} ${args})
    else()
      if(HPX_HAVE_PARCELPORT_IBVERBS)
        add_test(
          NAME "${category}.distributed.ibverbs.${name}"
          COMMAND ${cmd} "-p" "ibverbs" ${args})
      endif()
      if(HPX_HAVE_PARCELPORT_IPC)
        add_test(
          NAME "${category}.distributed.ipc.${name}"
          COMMAND ${cmd} "-p" "ipc" ${args})
      endif()
      if(HPX_HAVE_PARCELPORT_MPI)
        add_test(
          NAME "${category}.distributed.mpi.${name}"
          COMMAND ${cmd} "-p" "mpi" "-r" "mpi" ${args})
      endif()
      if(HPX_HAVE_PARCELPORT_TCP)
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

