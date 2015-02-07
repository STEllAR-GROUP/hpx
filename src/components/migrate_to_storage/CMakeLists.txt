# Copyright (c) 2007-2015 Hartmut Kaiser
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(root "${hpx_SOURCE_DIR}/hpx/components/migrate_to_storage")

add_definitions(-DHPX_MIGRATE_TO_STORAGE_MODULE_EXPORTS)

if(HPX_BUILD_DEFAULT_TARGETS)
  add_hpx_component(migrate_to_storage
    FOLDER "Core/Components/IO"
    HEADER_ROOT ${root}
    AUTOGLOB)
else()
  add_hpx_component(migrate_to_storage
    FOLDER "Core/Components/IO"
    HEADER_ROOT ${root}
    AUTOGLOB
    EXCLUDE_FROM_ALL)
endif()

add_hpx_pseudo_dependencies(
  components.migrate_to_storage
  migrate_to_storage_component)

add_hpx_pseudo_dependencies(
  core
  components.migrate_to_storage)

