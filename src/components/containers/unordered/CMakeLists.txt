# Copyright (c) 2015 Hartmut Kaiser
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

###############################################################################
set(root "${PROJECT_SOURCE_DIR}/hpx/components/containers/unordered")

add_hpx_component(unordered
  FOLDER "Core/Components/Containers"
  HEADER_ROOT ${root}
  AUTOGLOB
  ESSENTIAL)

add_hpx_pseudo_dependencies(components.containers_.unordered unordered_component)

