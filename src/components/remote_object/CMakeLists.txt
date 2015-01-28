# Copyright (c) 2007-2012 Hartmut Kaiser
# Copyright (c)      2011 Bryce Lelbach
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

###############################################################################
set(root "${hpx_SOURCE_DIR}/hpx/components/remote_object")

if(HPX_BUILD_DEFAULT_TARGETS)
  add_hpx_component(remote_object
    FOLDER "Core/Components"
    HEADER_ROOT ${root}
    AUTOGLOB)
endif()

add_hpx_pseudo_dependencies(components.remote_object
                            remote_object_component)
add_hpx_pseudo_dependencies(core
                          components.remote_object)

