# Copyright (c) 2019 The STE||AR-Group
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

# This file is auto generated. Please do not edit manually.

include(HPX_Message)
include(HPX_AddPseudoDependencies)
include(HPX_AddPseudoTarget)

set(HPX_LIBS
  algorithms
  assertion
  cache
  collectives
  config
  format
  hardware
  logging
  preprocessor
  testing
  timing
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
    add_custom_target(tests.headers.modules)
    add_hpx_pseudo_dependencies(tests.headers tests.headers.modules)
  endif()
endif()


hpx_info("Configuring modules:")

set(MODULE_FORCE_LINKING_INCLUDES)
set(MODULE_FORCE_LINKING_CALLS)
foreach(lib ${HPX_LIBS})
  add_subdirectory(${lib})

  set(MODULE_FORCE_LINKING_INCLUDES
    "${MODULE_FORCE_LINKING_INCLUDES}#include <hpx/${lib}/force_linking.hpp>
")

  set(MODULE_FORCE_LINKING_CALLS
    "${MODULE_FORCE_LINKING_CALLS}\n        ${lib}::force_linking();")
endforeach()

configure_file(
    "${PROJECT_SOURCE_DIR}/cmake/templates/modules.cpp.in"
    "${CMAKE_BINARY_DIR}/libs/modules.cpp"
    @ONLY)
