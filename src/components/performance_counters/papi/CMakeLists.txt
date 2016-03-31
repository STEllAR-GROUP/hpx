# Copyright (c) 2007-2011 Hartmut Kaiser
# Copyright (c)      2011 Bryce Lelbach
# Copyright (c)      2012 Maciej Brodowicz
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(root "${PROJECT_SOURCE_DIR}/hpx/components/performance_counters/papi_counters")

################################################################################
# PAPI event counters
################################################################################
if(HPX_WITH_PAPI)
  find_package(PAPI)

  if(NOT PAPI_FOUND)
    hpx_error("PAPI could not be found and HPX_WITH_PAPI=On, please specify PAPI_ROOT to point to the root of your PAPI installation")
  endif()

  include_directories(${PAPI_INCLUDE_DIR})

  if(HPX_WITH_DEFAULT_TARGETS)
    add_hpx_component(papi_counters
      FOLDER "Core/Components"
      HEADER_ROOT ${root}
      DEPENDENCIES ${PAPI_LIBRARIES}
      AUTOGLOB
      PLUGIN)
  endif()

  add_hpx_pseudo_dependencies(components.performance_counters_.papi
    papi_counters_component)
  add_hpx_pseudo_dependencies(core components.performance_counters_.papi)

endif()
