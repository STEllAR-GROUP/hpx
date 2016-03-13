# Copyright (c) 2007-2012 Hartmut Kaiser
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

###############################################################################
set(root "${PROJECT_SOURCE_DIR}/hpx/components/performance_counters/memory")

if(HPX_WITH_DEFAULT_TARGETS)
  add_hpx_component(memory
    FOLDER "Core/Components/Counters"
    HEADER_ROOT ${root}
    AUTOGLOB
    PLUGIN)
endif()

add_hpx_pseudo_dependencies(components.performance_counters.memory 
  memory_component)
add_hpx_pseudo_dependencies(core components.performance_counters.memory)

