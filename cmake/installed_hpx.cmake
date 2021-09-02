# Copyright (c) 2020 Hartmut Kaiser
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

find_package(HPX REQUIRED)

# ##############################################################################
hpx_local_option(
  HPXLocal_WITH_EXAMPLES BOOL "Build the HPX examples (default OFF)" OFF
  CATEGORY "Build Targets"
)
hpx_local_option(
  HPXLocal_WITH_TESTS BOOL "Build the HPX tests (default ON)" ON
  CATEGORY "Build Targets"
)
hpx_local_option(
  HPXLocal_WITH_TESTS_BENCHMARKS BOOL "Build HPX benchmark tests (default: ON)"
  ON ADVANCED CATEGORY "Build Targets"
)
hpx_local_option(
  HPXLocal_WITH_TESTS_REGRESSIONS BOOL
  "Build HPX regression tests (default: ON)" ON ADVANCED
  CATEGORY "Build Targets"
)
hpx_local_option(
  HPXLocal_WITH_TESTS_UNIT BOOL "Build HPX unit tests (default: ON)" ON ADVANCED
  CATEGORY "Build Targets"
)
hpx_local_option(
  HPXLocal_WITH_TESTS_HEADERS BOOL "Build HPX header tests (default: OFF)" OFF
  ADVANCED CATEGORY "Build Targets"
)
hpx_local_option(
  HPXLocal_WITH_TESTS_EXTERNAL_BUILD BOOL
  "Build external cmake build tests (default: ON)" ON ADVANCED
  CATEGORY "Build Targets"
)
hpx_local_option(
  HPXLocal_WITH_TESTS_EXAMPLES BOOL "Add HPX examples as tests (default: ON)" ON
  ADVANCED CATEGORY "Build Targets"
)

hpx_local_option(
  HPXLocal_WITH_COMPILE_ONLY_TESTS BOOL
  "Create build system support for compile time only HPX tests (default ON)" ON
  CATEGORY "Build Targets"
)
hpx_local_option(
  HPXLocal_WITH_FAIL_COMPILE_TESTS BOOL
  "Create build system support for fail compile HPX tests (default ON)" ON
  CATEGORY "Build Targets"
)

# disable all tests if HPXLocal_WITH_TESTS=OFF
if(NOT HPXLocal_WITH_TESTS)
  hpx_local_set_option(
    HPXLocal_WITH_TESTS_BENCHMARKS
    VALUE OFF
    FORCE
  )
  hpx_local_set_option(
    HPXLocal_WITH_TESTS_REGRESSIONS
    VALUE OFF
    FORCE
  )
  hpx_local_set_option(
    HPXLocal_WITH_TESTS_UNIT
    VALUE OFF
    FORCE
  )
  hpx_local_set_option(
    HPXLocal_WITH_TESTS_HEADERS
    VALUE OFF
    FORCE
  )
  hpx_local_set_option(
    HPXLocal_WITH_TESTS_EXTERNAL_BUILD
    VALUE OFF
    FORCE
  )
  hpx_local_set_option(
    HPXLocal_WITH_TESTS_EXAMPLES
    VALUE OFF
    FORCE
  )
endif()

if(HPXLocal_WITH_TESTS)
  # add pseudo targets
  hpx_local_add_pseudo_target(tests)
  if(HPXLocal_WITH_TESTS_UNIT)
    hpx_local_add_pseudo_target(tests.unit)
    hpx_local_add_pseudo_target(tests.unit.modules)
    hpx_local_add_pseudo_dependencies(tests tests.unit)
    hpx_local_add_pseudo_dependencies(tests.unit tests.unit.modules)
  endif()
  if(HPXLocal_WITH_TESTS_REGRESSIONS)
    hpx_local_add_pseudo_target(tests.regressions)
    hpx_local_add_pseudo_target(tests.regressions.components)
    hpx_local_add_pseudo_target(tests.regressions.modules)
    hpx_local_add_pseudo_dependencies(tests tests.regressions)
    hpx_local_add_pseudo_dependencies(
      tests.regressions tests.regressions.components
    )
    hpx_local_add_pseudo_dependencies(
      tests.regressions tests.regressions.modules
    )
  endif()
  if(HPXLocal_WITH_TESTS_BENCHMARKS)
    hpx_local_add_pseudo_target(tests.performance)
    hpx_local_add_pseudo_target(tests.performance.modules)
    hpx_local_add_pseudo_dependencies(tests tests.performance)
    hpx_local_add_pseudo_dependencies(
      tests.performance tests.performance.modules
    )
  endif()
  if(HPXLocal_WITH_TESTS_HEADERS)
    hpx_local_add_pseudo_target(tests.headers)
    hpx_local_add_pseudo_target(tests.headers.modules)
    hpx_local_add_pseudo_dependencies(tests tests.headers)
    hpx_local_add_pseudo_dependencies(tests.headers tests.headers.modules)
  endif()
  if(HPXLocal_WITH_EXAMPLES AND HPXLocal_WITH_TESTS_EXAMPLES)
    hpx_local_add_pseudo_target(tests.examples)
    hpx_local_add_pseudo_target(tests.examples.modules)
    hpx_local_add_pseudo_dependencies(tests tests.examples)
    hpx_local_add_pseudo_dependencies(tests.examples tests.examples.modules)
  endif()

  # enable cmake testing infrastructure
  enable_testing()
  include(CTest)

  # find Python interpreter (needed to run tests)
  find_package(PythonInterp)
  if(NOT PYTHONINTERP_FOUND)
    hpx_local_warn(
      "A python interpreter could not be found. The test suite can not be run automatically."
    )
  endif()

  # add actual tests, first iterate through all modules
  foreach(module ${HPXLocal_ENABLED_MODULES})
    if(EXISTS ${PROJECT_SOURCE_DIR}/libs/${lib}/${module}/tests)
      add_subdirectory(libs/${lib}/${module}/tests)
    endif()
  endforeach()

  # then main tests directory
  add_subdirectory(tests)
endif()

if(HPXLocal_WITH_EXAMPLES)
  # add pseudo targets
  hpx_local_add_pseudo_target(examples)
  hpx_local_add_pseudo_target(examples.modules)
  hpx_local_add_pseudo_dependencies(examples examples.modules)

  # add actual examples, iterate through all modules
  foreach(module ${HPXLocal_ENABLED_MODULES})
    if(EXISTS ${PROJECT_SOURCE_DIR}/libs/${lib}/${module}/examples)
      add_subdirectory(libs/${libs}/${module}/examples)
    endif()
  endforeach()

  add_subdirectory(examples)
endif()
