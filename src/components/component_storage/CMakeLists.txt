# Copyright (c) 2015 Hartmut Kaiser
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(root "${PROJECT_SOURCE_DIR}/hpx/components/component_storage")

add_definitions(-DHPX_MIGRATE_TO_STORAGE_MODULE_EXPORTS)

if(HPX_WITH_DEFAULT_TARGETS)
  add_hpx_component(component_storage
    DEPENDENCIES unordered_component
    FOLDER "Core/Components/IO"
    HEADER_ROOT ${root}
    AUTOGLOB)
else()
  add_hpx_component(component_storage
    DEPENDENCIES unordered_component
    FOLDER "Core/Components/IO"
    HEADER_ROOT ${root}
    AUTOGLOB
    EXCLUDE_FROM_ALL)
endif()

add_hpx_pseudo_dependencies(components.component_storage_
  component_storage_component)
add_hpx_pseudo_dependencies(core components.component_storage_)

