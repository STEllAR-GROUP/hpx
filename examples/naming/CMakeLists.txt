# Copyright (c) 2007-2009 Hartmut Kaiser
# Copyright (c) 2011      Bryce Lelbach
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying 
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

# define build targets for this directory
set(example_programs
    resolver_client
    resolver_server)

foreach(example_program ${example_programs})
  set(sources
      ${example_program}.cpp)

  source_group("Source Files" FILES ${sources})

  # add example executable
  add_hpx_executable(${example_program}
                     MODULE naming
                     SOURCES ${sources})

  # add a custom target for this example 
  add_custom_target(examples.naming.${example_program})

  # make pseudo-targets depend on master pseudo-target
  add_dependencies(examples.naming
                   examples.naming.${example_program})

  # add dependencies to pseudo-target 
  add_dependencies(examples.naming.${example_program}
                   ${example_program}_exe)
endforeach() 

