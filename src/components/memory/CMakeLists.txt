# Copyright (c) 2007-2012 Hartmut Kaiser
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

###############################################################################
set(root "${hpx_SOURCE_DIR}/hpx/components/memory")

if(HPX_DEFAULT_BUILD_TARGETS)
  add_hpx_component(memory
    FOLDER "Core/Components"
    HEADER_ROOT ${root}
    AUTOGLOB
    ESSENTIAL)
else()
  add_hpx_component(memory
    FOLDER "Core/Components"
    HEADER_ROOT ${root}
    AUTOGLOB)
endif()

add_hpx_pseudo_dependencies(components.memory
                            memory_component)

