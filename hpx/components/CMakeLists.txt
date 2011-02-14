# Copyright (c) 2007-2010 Hartmut Kaiser
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying 
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

# just recurse into all components subdirectories and execute the scripts there
set(hpx_COMPONENT_SUBDIRS 
    distributing_factory
   )

foreach(hpx_COMPONENT_SUBDIR ${hpx_COMPONENT_SUBDIRS})
    add_subdirectory(${hpx_COMPONENT_SUBDIR})
endforeach()

