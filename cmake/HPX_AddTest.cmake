# Copyright (c) 2011 Bryce Lelbach
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

function(add_hpx_test category name)
  set(options FAILURE_EXPECTED RUN_SERIAL)
  set(one_value_args EXECUTABLE LOCALITIES THREADS_PER_LOCALITY RUNWRAPPER)
  set(multi_value_args ARGS PARCELPORTS)
  cmake_parse_arguments(
    ${name} "${options}" "${one_value_args}" "${multi_value_args}" ${ARGN}
  )

  if(NOT ${name}_LOCALITIES)
    set(${name}_LOCALITIES 1)
  endif()

  if(NOT ${name}_THREADS_PER_LOCALITY)
    set(${name}_THREADS_PER_LOCALITY 1)
  elseif(HPX_WITH_TESTS_MAX_THREADS_PER_LOCALITY GREATER 0
         AND ${name}_THREADS_PER_LOCALITY GREATER
             HPX_WITH_TESTS_MAX_THREADS_PER_LOCALITY
  )
    set(${name}_THREADS_PER_LOCALITY ${HPX_WITH_TESTS_MAX_THREADS_PER_LOCALITY})
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

  if(${name}_RUN_SERIAL)
    set(run_serial TRUE)
  endif()

  # If --hpx:threads=cores or all
  if(${name}_THREADS_PER_LOCALITY LESS_EQUAL 0)
    set(run_serial TRUE)
  endif()

  set(args)

  foreach(arg ${${name}_UNPARSED_ARGUMENTS})
    set(args ${args} "${arg}")
  endforeach()
  set(args "-v" "--" ${args})
  if(${HPX_WITH_TESTS_DEBUG_LOG})
    set(args ${args}
             "--hpx:debug-hpx-log=${HPX_WITH_TESTS_DEBUG_LOG_DESTINATION}"
    )
  endif()

  if(${HPX_WITH_PARALLEL_TESTS_BIND_NONE}
     AND NOT run_serial
     AND NOT "${name}_RUNWRAPPER"
     AND (${name}_LOCALITIES STREQUAL "1" OR NOT ${HPX_WITH_NETWORKING})
  )
    set(args ${args} "--hpx:bind=none")
  endif()

  if(HPX_WITH_INSTALLED_VERSION)
    set(_script_location ${HPX_PREFIX})
  else()
    set(_script_location ${PROJECT_BINARY_DIR})
  endif()

  # cmake-format: off
  set(cmd
      "${PYTHON_EXECUTABLE}"
      "${_script_location}/bin/hpxrun.py"
      ${_exe}
        "-e" "${expected}"
        "-t" "${${name}_THREADS_PER_LOCALITY}"
  )
  # cmake-format: on

  # if runwrapper is set we don't want to use HPX parcelports
  if(${name}_RUNWRAPPER)
    list(
      APPEND
      cmd
      "-r"
      "${${name}_RUNWRAPPER}"
      "-l"
      "${${name}_LOCALITIES}"
      "-p"
      "none"
    )
    set(${name}_LOCALITIES "1")
  elseif(HPX_WITH_NETWORKING)
    list(APPEND cmd "-l" "${${name}_LOCALITIES}")
  else()
    set(${name}_LOCALITIES "1")
  endif()

  if(${name}_LOCALITIES STREQUAL "1")
    set(_full_name "${category}.${name}")
    add_test(NAME "${category}.${name}" COMMAND ${cmd} ${args})
    if(${run_serial})
      set_tests_properties("${_full_name}" PROPERTIES RUN_SERIAL TRUE)
    endif()
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
        set(_full_name "${category}.distributed.verbs.${name}")
        add_test(NAME "${_full_name}" COMMAND ${cmd} "-p" "verbs" ${args})
        set_tests_properties("${_full_name}" PROPERTIES RUN_SERIAL TRUE)
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
        set(_full_name "${category}.distributed.mpi.${name}")
        add_test(NAME "${_full_name}" COMMAND ${cmd} "-p" "mpi" "-r" "mpi"
                                              ${args}
        )
        set_tests_properties("${_full_name}" PROPERTIES RUN_SERIAL TRUE)
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
        set(_full_name "${category}.distributed.tcp.${name}")
        add_test(NAME "${_full_name}" COMMAND ${cmd} "-p" "tcp" ${args})
        set_tests_properties("${_full_name}" PROPERTIES RUN_SERIAL TRUE)
      endif()
    endif()
  endif()

endfunction(add_hpx_test)

function(add_hpx_test_target_dependencies category name)
  set(one_value_args PSEUDO_DEPS_NAME)
  cmake_parse_arguments(
    ${name} "${options}" "${one_value_args}" "${multi_value_args}" ${ARGN}
  )
  # default target_extension is _test but for examples.* target, it may vary
  if(NOT ("${category}" MATCHES "tests.examples*"))
    set(_ext "_test")
  endif()
  # Add a custom target for this example
  add_hpx_pseudo_target(${category}.${name})
  # Make pseudo-targets depend on master pseudo-target
  add_hpx_pseudo_dependencies(${category} ${category}.${name})
  # Add dependencies to pseudo-target
  if(${name}_PSEUDO_DEPS_NAME)
    # When the test depend on another executable name
    add_hpx_pseudo_dependencies(
      ${category}.${name} ${${name}_PSEUDO_DEPS_NAME}${_ext}
    )
  else()
    add_hpx_pseudo_dependencies(${category}.${name} ${name}${_ext})
  endif()
endfunction(add_hpx_test_target_dependencies)

# To add test to the category root as in tests/regressions/ with correct name
function(add_test_and_deps_test category subcategory name)
  if("${subcategory}" STREQUAL "")
    add_hpx_test(tests.${category} ${name} ${ARGN})
    add_hpx_test_target_dependencies(tests.${category} ${name} ${ARGN})
  else()
    add_hpx_test(tests.${category}.${subcategory} ${name} ${ARGN})
    add_hpx_test_target_dependencies(
      tests.${category}.${subcategory} ${name} ${ARGN}
    )
  endif()
endfunction(add_test_and_deps_test)

function(add_hpx_unit_test subcategory name)
  add_test_and_deps_test("unit" "${subcategory}" ${name} ${ARGN})
endfunction(add_hpx_unit_test)

function(add_hpx_regression_test subcategory name)
  # ARGN needed in case we add a test with the same executable
  add_test_and_deps_test("regressions" "${subcategory}" ${name} ${ARGN})
endfunction(add_hpx_regression_test)

function(add_hpx_performance_test subcategory name)
  add_test_and_deps_test(
    "performance" "${subcategory}" ${name} ${ARGN} RUN_SERIAL
  )
endfunction(add_hpx_performance_test)

function(add_hpx_example_test subcategory name)
  add_test_and_deps_test("examples" "${subcategory}" ${name} ${ARGN})
endfunction(add_hpx_example_test)

# To create target examples.<name> when calling make examples need 2 distinct
# rules for examples and tests.examples
function(add_hpx_example_target_dependencies subcategory name)
  set(options DEPS_ONLY)
  cmake_parse_arguments(
    ${name} "${options}" "${one_value_args}" "${multi_value_args}" ${ARGN}
  )
  if(NOT ${name}_DEPS_ONLY)
    # Add a custom target for this example
    add_hpx_pseudo_target(examples.${subcategory}.${name})
  endif()
  # Make pseudo-targets depend on master pseudo-target
  add_hpx_pseudo_dependencies(
    examples.${subcategory} examples.${subcategory}.${name}
  )
  # Add dependencies to pseudo-target
  add_hpx_pseudo_dependencies(examples.${subcategory}.${name} ${name})
endfunction(add_hpx_example_target_dependencies)
