# Copyright (c) 2015 Hartmut Kaiser
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

###############################################################################
set(root "${PROJECT_SOURCE_DIR}/hpx/components/containers/partitioned_vector")

add_hpx_component(partitioned_vector
  FOLDER "Core/Components/Containers"
  HEADER_ROOT ${root}
  AUTOGLOB
  ESSENTIAL)

add_hpx_pseudo_dependencies(
    components.containers.partitioned_vector partitioned_vector_component)

