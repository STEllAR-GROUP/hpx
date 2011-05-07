# Copyright (c) 2011 Bryce Lelbach
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying 
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

if(NOT MSVC)
  set(example_programs
      implicit_bootstrap_constants_v1
      implicit_bootstrap_constants_v2)
  
  foreach(example_program ${example_programs})
    set(sources ${example_program}.cpp)
  
    source_group("Source Files" FILES ${sources})
  
    # add example executable
    add_hpx_executable(${example_program}
                       MODULE agas
                       DEPENDENCIES agas_stdmap_component
                                    agas_stdmap_tcpip_component
                       SOURCES ${sources})
  
    # add a custom target for this example 
    add_hpx_pseudo_target(examples.agas.${example_program})
  
    # make pseudo-targets depend on master pseudo-target
    add_hpx_pseudo_dependencies(examples.agas
                                examples.agas.${example_program})
  
    # add dependencies to pseudo-target 
    add_hpx_pseudo_dependencies(examples.agas.${example_program}
                                ${example_program}_exe)
  endforeach() 
endif()

