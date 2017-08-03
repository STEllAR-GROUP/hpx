# Copyright (c) 2015-2017 Hartmut Kaiser
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

###############################################################################
set(root "${PROJECT_SOURCE_DIR}/hpx/components/containers/partitioned_vector")

add_definitions(-DHPX_PARTITIONED_VECTOR_MODULE_EXPORTS)

add_hpx_component(partitioned_vector
  FOLDER "Core/Components/Containers"
  HEADER_ROOT ${root}
  AUTOGLOB
  ESSENTIAL)

add_hpx_pseudo_dependencies(
    components.containers_.partitioned_vector partitioned_vector_component)

