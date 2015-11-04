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
      if(HPX_WITH_PARCELPORT_IBVERBS)
        set(_add_test FALSE)
        if(DEFINED ${name}_PARCELPORTS)
          set(PP_FOUND -1)
          list(FIND ${name}_PARCELPORTS "ibverbs" PP_FOUND)
          if(NOT PP_FOUND EQUAL -1)
            set(_add_test TRUE)
          endif()
        else()
          set(_add_test TRUE)
        endif()
        if(_add_test)
          add_test(
            NAME "${category}.distributed.ibverbs.${name}"
            COMMAND ${cmd} "-p" "ibverbs" ${args})
        endif()
      endif()
      if(HPX_WITH_PARCELPORT_IPC)
        set(_add_test FALSE)
        if(DEFINED ${name}_PARCELPORTS)
          set(PP_FOUND -1)
          list(FIND ${name}_PARCELPORTS "ipc" PP_FOUND)
          if(NOT PP_FOUND EQUAL -1)
            set(_add_test TRUE)
          endif()
        else()
          set(_add_test TRUE)
        endif()
        if(_add_test)
          add_test(
            NAME "${category}.distributed.ipc.${name}"
            COMMAND ${cmd} "-p" "ipc" ${args})
        endif()
      endif()
      if(HPX_WITH_PARCELPORT_MPI)
        set(_add_test FALSE)
        if(DEFINED ${name}_PARCELPORTS)
          set(PP_FOUND -1)
          list(FIND ${name}_PARCELPORTS "mpi" PP_FOUND)
          if(NOT PP_FOUND EQUAL -1)
            set(_add_test TRUE)
          endif()
        else()
          set(_add_test TRUE)
        endif()
        if(_add_test)
          add_test(
            NAME "${category}.distributed.mpi.${name}"
            COMMAND ${cmd} "-p" "mpi" "-r" "mpi" ${args})
        endif()
      endif()
      if(HPX_WITH_PARCELPORT_TCP)
        set(_add_test FALSE)
        if(DEFINED ${name}_PARCELPORTS)
          set(PP_FOUND -1)
          list(FIND ${name}_PARCELPORTS "tcp" PP_FOUND)
          if(NOT PP_FOUND EQUAL -1)
            set(_add_test TRUE)
          endif()
        else()
          set(_add_test TRUE)
        endif()
        if(_add_test)
          add_test(
            NAME "${category}.distributed.tcp.${name}"
            COMMAND ${cmd} "-p" "tcp" ${args})
        endif()
      endif()
    endif()
endmacro()

macro(add_hpx_unit_test category name)
  add_hpx_test("tests.unit.${category}" ${name} ${ARGN})
endmacro()

macro(add_hpx_regression_test category name)
  add_hpx_test("tests.regressions.${category}" ${name} ${ARGN})
endmacro()

