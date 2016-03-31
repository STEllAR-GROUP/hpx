# Copyright (c) 2007-2012 Hartmut Kaiser
# Copyright (c)      2011 Bryce Lelbach
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(root "${PROJECT_SOURCE_DIR}/hpx/components/iostreams")

add_definitions(-DHPX_IOSTREAMS_MODULE_EXPORTS)

if(HPX_WITH_DEFAULT_TARGETS)
  add_hpx_component(iostreams
    FOLDER "Core/Components/IO"
    HEADER_ROOT ${root}
    AUTOGLOB)
else()
  add_hpx_component(iostreams
    FOLDER "Core/Components/IO"
    HEADER_ROOT ${root}
    AUTOGLOB
    EXCLUDE_FROM_ALL)
endif()

add_hpx_pseudo_dependencies(components.iostreams_ iostreams_component)
add_hpx_pseudo_dependencies(core components.iostreams_)

