# Copyright (c) 2011 Bryce Lelbach
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

function(add_hpx_test category name)
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

  if(TARGET ${${name}_EXECUTABLE}_test)
    set(_exe "$<TARGET_FILE:${${name}_EXECUTABLE}_test>")
  elseif(TARGET ${${name}_EXECUTABLE})
    set(_exe "$<TARGET_FILE:${${name}_EXECUTABLE}>")
  else()
    set(_exe "${${name}_EXECUTABLE}")
  endif()

  set(expected "0")

  if(${name}_FAILURE_EXPECTED)
    set(expected "1")
  endif()

  set(args)

  foreach(arg ${${name}_UNPARSED_ARGUMENTS})
    set(args ${args} "${arg}")
  endforeach()
  set(args "-v" "--" ${args})

  set(cmd "${PYTHON_EXECUTABLE}"
          "${CMAKE_BINARY_DIR}/bin/hpxrun.py"
          ${_exe}
          "-e" "${expected}"
          "-t" "${${name}_THREADS_PER_LOCALITY}")

  if(HPX_WITH_NETWORKING)
      list(APPEND cmd "-l" "${${name}_LOCALITIES}")
  else()
      set(${name}_LOCALITIES "1")
  endif()

  if(${name}_LOCALITIES STREQUAL "1")
    add_test(
      NAME "${category}.${name}"
      COMMAND ${cmd} ${args})
    else()
      if(HPX_WITH_PARCELPORT_VERBS)
        set(_add_test FALSE)
        if(DEFINED ${name}_PARCELPORTS)
          set(PP_FOUND -1)
          list(FIND ${name}_PARCELPORTS "verbs" PP_FOUND)
          if(NOT PP_FOUND EQUAL -1)
            set(_add_test TRUE)
          endif()
        else()
          set(_add_test TRUE)
        endif()
        if(_add_test)
          add_test(
            NAME "${category}.distributed.verbs.${name}"
            COMMAND ${cmd} "-p" "verbs" ${args})
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
endfunction()

function(add_hpx_unit_test category name)
  add_hpx_test("tests.unit.${category}" ${name} ${ARGN})
endfunction()

function(add_hpx_regression_test category name)
  add_hpx_test("tests.regressions.${category}" ${name} ${ARGN})
endfunction()

function(add_hpx_example_test category name)
  add_hpx_test("tests.examples.${category}" ${name} ${ARGN})
endfunction()


