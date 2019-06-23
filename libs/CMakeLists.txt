# Copyright (c) 2019 The STE||AR-Group
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

# This file is auto generated. Please do not edit manually.

include(HPX_Message)

set(HPX_LIBS
  cache
  collectives
  config
  preprocessor
  CACHE INTERNAL "" FORCE
)

if (HPX_WITH_TESTS_UNIT)
  add_hpx_pseudo_target(tests.unit.modules)
endif()

if (HPX_WITH_TESTS_REGRESSIONS)
  add_hpx_pseudo_target(tests.regressions.modules)
endif()

if (HPX_WITH_TESTS_BENCHMARKS)
  add_hpx_pseudo_target(tests.performance.modules)
endif()

if (HPX_WITH_TESTS_HEADERS)
  add_custom_target(tests.headers.modules)
endif()

hpx_info("Configuring modules:")

foreach(lib ${HPX_LIBS})
  add_subdirectory(${lib})
endforeach()

