# Copyright (c) 2019 The STE||AR-Group
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

include(HPX_AddPseudoDependencies)
include(HPX_AddPseudoTarget)

set(HPX_COMPONENTS
  component_storage
  containers
  iostreams
  performance_counters
  process
  CACHE INTERNAL "list of HPX components" FORCE
)

hpx_info("Configuring components:")

add_hpx_pseudo_target(components)

foreach(component ${HPX_COMPONENTS})
  add_hpx_pseudo_target(components.${component})
  add_subdirectory(${component})
  add_hpx_pseudo_dependencies(components components.${component})
endforeach()

add_hpx_pseudo_dependencies(core components)
