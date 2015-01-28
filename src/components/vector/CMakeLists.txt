# Copyright (c) 2014 Hartmut Kaiser
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

###############################################################################
set(root "${hpx_SOURCE_DIR}/hpx/components/vector")

add_hpx_component(vector
  FOLDER "Core/Components"
  HEADER_ROOT ${root}
  AUTOGLOB
  ESSENTIAL)

add_hpx_pseudo_dependencies(components.vector
                            vector_component)

