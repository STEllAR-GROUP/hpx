# Copyright (c) 2014 John Biddiscombe
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(example_programs
   )

if(Boost_RANDOM_FOUND)
  set(example_programs
      ${example_programs}
      rnd_future_reduce)
  set(rnd_future_reduce_FLAGS DEPENDENCIES iostreams_component)
endif()

foreach(example_program ${example_programs})
  set(sources
      ${example_program}.cpp)

  source_group("Source Files" FILES ${sources})

  # add example executable
  add_hpx_executable(${example_program}
                     SOURCES ${sources}
                     ${${example_program}_FLAGS}
                     FOLDER "Examples/FutureReduce")

  # add a custom target for this example
  add_hpx_pseudo_target(examples.future_reduce.${example_program})

  # make pseudo-targets depend on master pseudo-target
  add_hpx_pseudo_dependencies(examples.future_reduce
                              examples.future_reduce.${example_program})

  # add dependencies to pseudo-target
  add_hpx_pseudo_dependencies(examples.future_reduce.${example_program}
                              ${example_program}_exe)
endforeach()

