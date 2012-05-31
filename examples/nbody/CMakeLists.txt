# Copyright (c) 2007-2009 Hartmut Kaiser
# Copyright (c) 2011      Bryce Lelbach
# Copyright (c) 2012      Daniel Kogler
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

add_subdirectory(nbody)

set(examples
    global_solve
   )

# for all targets specified above
foreach(example ${examples})
  set(sources ${example}.cpp)

  source_group("Source Files" FILES ${sources})

  # add example executable
  add_hpx_executable(${example}
    SOURCES ${sources}
    DEPENDENCIES ${example}_component)

  # add a custom target for this example
  add_hpx_pseudo_target(examples.nbody.${example})

  # make pseudo-targets depend on master pseudo-target
  add_hpx_pseudo_dependencies(examples.nbody
                              examples.nbody.${example})

  # add dependencies to pseudo-target
  add_hpx_pseudo_dependencies(examples.nbody.${example}
                              ${example}_exe)
endforeach()

