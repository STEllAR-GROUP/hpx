# Copyright (c) 2007-2016 Hartmut Kaiser
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(root "${PROJECT_SOURCE_DIR}/hpx/components/process")

add_definitions(-DHPX_PROCESS_EXPORTS)

add_hpx_component(process
    FOLDER "Core/Components/Process"
    HEADER_ROOT ${root}
    AUTOGLOB)

add_hpx_pseudo_dependencies(components.process process_component)
add_hpx_pseudo_dependencies(core components.process)

