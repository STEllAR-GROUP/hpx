# Copyright (c) 2007-2009 Hartmut Kaiser
# Copyright (c) 2011      Bryce Lelbach
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

add_subdirectory(central_tuplespace)

set(examples
    simple_central_tuplespace
   )

# for all targets specified above
foreach(example ${examples})
  set(sources ${example}_client.cpp)

  source_group("Source Files" FILES ${sources})

  # add example executable
  add_hpx_executable(${example}_client
    SOURCES ${sources}
    DEPENDENCIES ${example}_component iostreams_component
    FOLDER "Examples/TupleSpace/${example}")

  # add a custom target for this example
  add_hpx_pseudo_target(examples.tuplespace.${example})

  # make pseudo-targets depend on master pseudo-target
  add_hpx_pseudo_dependencies(examples.tuplespace
                              examples.tuplespace.${example})

  # add dependencies to pseudo-target
  add_hpx_pseudo_dependencies(examples.tuplespace.${example}
                              ${example}_client_exe)
endforeach()

