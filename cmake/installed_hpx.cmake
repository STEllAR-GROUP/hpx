# Copyright (c) 2020 Hartmut Kaiser
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

find_package(HPX REQUIRED)

# ##############################################################################
hpx_option(
  HPX_WITH_EXAMPLES BOOL "Build the HPX examples (default OFF)" OFF
  CATEGORY "Build Targets"
)
hpx_option(
  HPX_WITH_TESTS BOOL "Build the HPX tests (default ON)" ON
  CATEGORY "Build Targets"
)
hpx_option(
  HPX_WITH_TESTS_BENCHMARKS BOOL "Build HPX benchmark tests (default: ON)" ON
  ADVANCED CATEGORY "Build Targets"
)
hpx_option(
  HPX_WITH_TESTS_REGRESSIONS BOOL "Build HPX regression tests (default: ON)" ON
  ADVANCED CATEGORY "Build Targets"
)
hpx_option(
  HPX_WITH_TESTS_UNIT BOOL "Build HPX unit tests (default: ON)" ON ADVANCED
  CATEGORY "Build Targets"
)
hpx_option(
  HPX_WITH_TESTS_HEADERS BOOL "Build HPX header tests (default: OFF)" OFF
  ADVANCED CATEGORY "Build Targets"
)
hpx_option(
  HPX_WITH_TESTS_EXTERNAL_BUILD BOOL
  "Build external cmake build tests (default: ON)" ON ADVANCED
  CATEGORY "Build Targets"
)
hpx_option(
  HPX_WITH_TESTS_EXAMPLES BOOL "Add HPX examples as tests (default: ON)" ON
  ADVANCED CATEGORY "Build Targets"
)

hpx_option(
  HPX_WITH_COMPILE_ONLY_TESTS BOOL
  "Create build system support for compile time only HPX tests (default ON)" ON
  CATEGORY "Build Targets"
)
hpx_option(
  HPX_WITH_FAIL_COMPILE_TESTS BOOL
  "Create build system support for fail compile HPX tests (default ON)" ON
  CATEGORY "Build Targets"
)

# disable all tests if HPX_WITH_TESTS=OFF
if(NOT HPX_WITH_TESTS)
  hpx_set_option(
    HPX_WITH_TESTS_BENCHMARKS
    VALUE OFF
    FORCE
  )
  hpx_set_option(
    HPX_WITH_TESTS_REGRESSIONS
    VALUE OFF
    FORCE
  )
  hpx_set_option(
    HPX_WITH_TESTS_UNIT
    VALUE OFF
    FORCE
  )
  hpx_set_option(
    HPX_WITH_TESTS_HEADERS
    VALUE OFF
    FORCE
  )
  hpx_set_option(
    HPX_WITH_TESTS_EXTERNAL_BUILD
    VALUE OFF
    FORCE
  )
  hpx_set_option(
    HPX_WITH_TESTS_EXAMPLES
    VALUE OFF
    FORCE
  )
endif()

if(HPX_WITH_TESTS)
  # add pseudo targets
  add_hpx_pseudo_target(tests)
  if(HPX_WITH_TESTS_UNIT)
    add_hpx_pseudo_target(tests.unit)
    add_hpx_pseudo_target(tests.unit.modules)
    add_hpx_pseudo_dependencies(tests tests.unit)
    add_hpx_pseudo_dependencies(tests.unit tests.unit.modules)
  endif()
  if(HPX_WITH_TESTS_REGRESSIONS)
    add_hpx_pseudo_target(tests.regressions)
    add_hpx_pseudo_target(tests.regressions.components)
    add_hpx_pseudo_target(tests.regressions.modules)
    add_hpx_pseudo_dependencies(tests tests.regressions)
    add_hpx_pseudo_dependencies(tests.regressions tests.regressions.components)
    add_hpx_pseudo_dependencies(tests.regressions tests.regressions.modules)
  endif()
  if(HPX_WITH_TESTS_BENCHMARKS)
    add_hpx_pseudo_target(tests.performance)
    add_hpx_pseudo_target(tests.performance.modules)
    add_hpx_pseudo_dependencies(tests tests.performance)
    add_hpx_pseudo_dependencies(tests.performance tests.performance.modules)
  endif()
  if(HPX_WITH_TESTS_HEADERS)
    add_hpx_pseudo_target(tests.headers)
    add_hpx_pseudo_target(tests.headers.modules)
    add_hpx_pseudo_dependencies(tests tests.headers)
    add_hpx_pseudo_dependencies(tests.headers tests.headers.modules)
  endif()
  if(HPX_WITH_EXAMPLES AND HPX_WITH_TESTS_EXAMPLES)
    add_hpx_pseudo_target(tests.examples)
    add_hpx_pseudo_target(tests.examples.modules)
    add_hpx_pseudo_dependencies(tests tests.examples)
    add_hpx_pseudo_dependencies(tests.examples tests.examples.modules)
  endif()

  # enable cmake testing infrastructure
  enable_testing()
  include(CTest)

  # find Python interpreter (needed to run tests)
  find_package(PythonInterp)
  if(NOT PYTHONINTERP_FOUND)
    hpx_warn(
      "A python interpreter could not be found. The test suite can not be run automatically."
    )
  endif()

  # add actual tests, first iterate through all modules
  foreach(module ${HPX_FULL_ENABLED_MODULES})
    if(EXISTS ${PROJECT_SOURCE_DIR}/libs/full/${module}/tests)
      add_subdirectory(libs/full/${module}/tests)
    endif()
  endforeach()

  # then main tests directory
  add_subdirectory(tests)
endif()

if(HPX_WITH_EXAMPLES)
  # add pseudo targets
  add_hpx_pseudo_target(examples)
  add_hpx_pseudo_target(examples.modules)
  add_hpx_pseudo_dependencies(examples examples.modules)

  # add actual examples, iterate through all modules
  foreach(lib core full parallelism)
    string(TOUPPER ${lib} lib_upper)
    foreach(module ${HPX_${lib_upper}_ENABLED_MODULES})
      if(EXISTS ${PROJECT_SOURCE_DIR}/libs/${lib}/${module}/examples)
        add_subdirectory(libs/${libs}/${module}/examples)
      endif()
    endforeach()
  endforeach()

  add_subdirectory(examples)
endif()
