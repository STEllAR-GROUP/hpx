# Copyright (c) 2007-2009 Hartmut Kaiser
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying 
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

# just recurse into all components subdirectories and execute the scripts there
set(component_SUDIRS 
    #distributed_set
    #distributed_map
    #graph
    #distributing_factory
   )

foreach(subdir ${component_SUDIRS})
    add_subdirectory(${subdir})
endforeach()
