# Copyright (c) 2011      Thomas Heller
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying 
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(bright_future_programs
        root_finder
        vector_root_finder
        gs_single
        gs_hpx
        gs_hpx_blocked
   )


add_hpx_component(
    gs_hpx
    MODULE bright_future
    SOURCES server/remote_lse.cpp
    HEADERS server/remote_lse.hpp grid.hpp
    DEPENDENCIES iostreams_component distributing_factory_component hpx_serialization ${BOOST_serialization_LIBRARY}
)

set(gs_single_sources gs.cpp)
set(gs_hpx_sources gs.cpp)
set(gs_hpx_blocked_sources gs.cpp)

set(vector_root_finder_FLAGS DEPENDENCIES iostreams_component)
set(gs_single_FLAGS DEPENDENCIES iostreams_component)
set(gs_hpx_FLAGS DEPENDENCIES gs_hpx_component iostreams_component hpx_serialization ${BOOST_serialization_LIBRARY})
set(gs_hpx_blocked_FLAGS DEPENDENCIES gs_hpx_component iostreams_component hpx_serialization ${BOOST_serialization_LIBRARY})

foreach(bright_future_program ${bright_future_programs})
  set(sources
       ${${bright_future_program}_sources} ${bright_future_program}.cpp)

  source_group("Source Files" FILES ${sources})

  # add bright_future executable
  add_hpx_executable(${bright_future_program}
                     MODULE bright_future
                     SOURCES ${sources}
                     ${${bright_future_program}_FLAGS})

  # add a custom target for this example 
  add_hpx_pseudo_target(examples.bright_future.${bright_future_program})

  # make pseudo-targets depend on master pseudo-target
  add_hpx_pseudo_dependencies(examples.bright_future
                              examples.bright_future.${bright_future_program})

  # add dependencies to pseudo-target 
  add_hpx_pseudo_dependencies(examples.bright_future.${bright_future_program}
                              ${bright_future_program}_exe)
endforeach()

set_target_properties(gs_single_exe PROPERTIES COMPILE_DEFINITIONS BRIGHT_FUTURE_NO_HPX)
set_target_properties(gs_single_exe PROPERTIES COMPILE_FLAGS -fopenmp)
set_target_properties(gs_single_exe PROPERTIES LINK_FLAGS -fopenmp)

#add_hpx_executable(
#    gs_hpx
#    MODULE bright_future
#    SOURCES gs.cpp
#    DEPENDENCIES iostreams_component gs_hpx_component
#)

# add a custom target for this example 
#add_hpx_pseudo_target(examples.bright_future.gs_hpx)
    
# make pseudo-targets depend on master pseudo-target
#add_hpx_pseudo_dependencies(examples.bright_future
#        examples.bright_future.gs_hpx)

# add dependencies to pseudo-target 
#add_hpx_pseudo_dependencies(examples.bright_future.gs_hpx
#                              gs_hpx_exe)
