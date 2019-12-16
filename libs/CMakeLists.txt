# Copyright (c) 2019 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

# This file is auto generated. Please do not edit manually.

include(HPX_Message)
include(HPX_AddPseudoDependencies)
include(HPX_AddPseudoTarget)

set(HPX_LIBS
  affinity
  algorithms
  allocator_support
  assertion
  basic_execution
  batch_environments
  cache
  checkpoint
  collectives
  compute
  compute_cuda
  concepts
  concurrency
  config
  coroutines
  datastructures
  debugging
  errors
  execution
  filesystem
  format
  functional
  hardware
  hashing
  iterator_support
  local_lcos
  logging
  memory
  mpi
  naming_base
  plugin
  preprocessor
  program_options
  resiliency
  resource_partitioner
  segmented_algorithms
  serialization
  static_reinit
  statistics
  synchronization
  testing
  threading_base
  thread_support
  threadmanager
  timing
  topology
  type_support
  util
  CACHE INTERNAL "list of HPX modules" FORCE
)


# add example pseudo targets needed for modules
if(HPX_WITH_EXAMPLES)
  add_hpx_pseudo_target(examples.modules)
  add_hpx_pseudo_dependencies(examples examples.modules)
endif()

# add test pseudo targets needed for modules
if(HPX_WITH_TESTS)
  if (HPX_WITH_TESTS_UNIT)
    add_hpx_pseudo_target(tests.unit.modules)
    add_hpx_pseudo_dependencies(tests.unit tests.unit.modules)
  endif()

  if (HPX_WITH_EXAMPLES AND HPX_WITH_TESTS_EXAMPLES)
    add_hpx_pseudo_target(tests.examples.modules)
    add_hpx_pseudo_dependencies(tests.examples tests.examples.modules)
  endif()

  if (HPX_WITH_TESTS_REGRESSIONS)
    add_hpx_pseudo_target(tests.regressions.modules)
    add_hpx_pseudo_dependencies(tests.regressions tests.regressions.modules)
  endif()

  if (HPX_WITH_TESTS_BENCHMARKS)
    add_hpx_pseudo_target(tests.performance.modules)
    add_hpx_pseudo_dependencies(tests.performance tests.performance.modules)
  endif()

  if (HPX_WITH_TESTS_HEADERS)
    add_hpx_pseudo_target(tests.headers.modules)
    add_hpx_pseudo_dependencies(tests.headers tests.headers.modules)
  endif()
endif()


hpx_info("")
hpx_info("Configuring modules:")

# variables needed for modules.cpp
set(MODULE_FORCE_LINKING_INCLUDES)
set(MODULE_FORCE_LINKING_CALLS)

# variables needed for config_strings_modules.hpp
set(CONFIG_STRINGS_MODULES_INCLUDES)
set(CONFIG_STRINGS_MODULES_ENTRIES)

foreach(lib ${HPX_LIBS})
  add_subdirectory(${lib})

  set(MODULE_FORCE_LINKING_INCLUDES
    "${MODULE_FORCE_LINKING_INCLUDES}#include <hpx/${lib}/force_linking.hpp>\n")

  set(MODULE_FORCE_LINKING_CALLS
    "${MODULE_FORCE_LINKING_CALLS}\n        ${lib}::force_linking();")

  set(CONFIG_STRINGS_MODULES_INCLUDES
    "${CONFIG_STRINGS_MODULES_INCLUDES}#include <hpx/${lib}/config/config_strings.hpp>\n")
  set(CONFIG_STRINGS_MODULES_ENTRIES
    "${CONFIG_STRINGS_MODULES_ENTRIES}\n        { \"${lib}\", ${lib}::config_strings },")
endforeach()

configure_file(
    "${PROJECT_SOURCE_DIR}/cmake/templates/modules.cpp.in"
    "${PROJECT_BINARY_DIR}/libs/modules.cpp"
    @ONLY)

configure_file(
  "${PROJECT_SOURCE_DIR}/cmake/templates/config_defines_strings_modules.hpp.in"
  "${PROJECT_BINARY_DIR}/hpx/config/config_defines_strings_modules.hpp"
  @ONLY)
configure_file(
  "${PROJECT_SOURCE_DIR}/cmake/templates/config_defines_strings_modules.hpp.in"
  "${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/hpx/config/config_defines_strings_modules.hpp"
  @ONLY)
