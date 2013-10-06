# Copyright (c) 2007-2011 Hartmut Kaiser
# Copyright (c)      2011 Bryce Lelbach
# Copyright (c)      2012 Maciej Brodowicz
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(root "${hpx_SOURCE_DIR}/hpx/components/papi_counters")

include_directories(${PAPI_INCLUDE_DIR})

if(HPX_DEFAULT_BUILD_TARGETS)
  add_hpx_component(papi_counters
    FOLDER "Core/Components"
    HEADER_ROOT ${root}
    DEPENDENCIES ${PAPI_LIBRARY}
    AUTOGLOB
    ESSENTIAL)
else()
  add_hpx_component(papi_counters
    FOLDER "Core/Components"
    HEADER_ROOT ${root}
    DEPENDENCIES ${PAPI_LIBRARY}
    AUTOGLOB)
endif()

add_hpx_pseudo_dependencies(
  components.papi_counters papi_counters_component)

