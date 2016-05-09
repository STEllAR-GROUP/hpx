# Copyright (c) 2016 Thomas Heller
# Copyright (c) 2016 Hartmut Kaiser
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(example_programs)

if(HPX_WITH_CUDA)
  set(example_programs ${example_programs}
      data_copy
      hello_compute
    )
  set(data_copy_CUDA On)
  set(hello_compute_CUDA On)
endif()

foreach(example_program ${example_programs})
  if(${${example_program}_CUDA})
    set(sources
      ${example_program}.cu)
  else()
    set(sources
        ${example_program}.cpp)
  endif()

  source_group("Source Files" FILES ${sources})

  # add example executable
  add_hpx_executable(${example_program}
                     SOURCES ${sources}
                     ${${example_program}_FLAGS}
                     FOLDER "Examples/Compute")

  # add a custom target for this example
  add_hpx_pseudo_target(examples.compute.${example_program})

  # make pseudo-targets depend on master pseudo-target
  add_hpx_pseudo_dependencies(examples.compute
                              examples.compute.${example_program})

  # add dependencies to pseudo-target
  add_hpx_pseudo_dependencies(examples.compute.${example_program}
                              ${example_program}_exe)
endforeach()

