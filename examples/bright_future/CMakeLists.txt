# Copyright (c) 2011      Thomas Heller
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

add_definitions(-DHPX_ACTION_ARGUMENT_LIMIT=10)
add_definitions(-DHPX_COMPONENT_CREATE_ARG_MAX=10)

set(bright_future_programs
        root_finder
        vector_root_finder
        gs_single
#        gs_hpx
        gs_hpx_blocked
        gs_hpx_grid_ro
        remote_functions
        remote_object
   )

set(bright_future_programs ${bright_future_programs}
    dataflow_test
    gs_hpx_blocked_df
    gs_hpx_distributed
   )

add_hpx_component(
    gs_hpx
    SOURCES server/remote_lse.cpp
    HEADERS server/remote_lse.hpp grid.hpp
    DEPENDENCIES iostreams_component distributing_factory_component ${GOOGLE_PERFTOOLS_LIBRARY}
)

add_hpx_component(
    bright_future_dataflow
    SOURCES dataflow/dataflow_component.cpp
    HEADERS dataflow/dataflow.hpp
    DEPENDENCIES iostreams_component ${GOOGLE_PERFTOOLS_LIBRARY}
)

set(gs_single_sources gs.cpp)
#set(gs_hpx_sources gs.cpp)
set(gs_hpx_blocked_sources gs.cpp)
set(gs_hpx_blocked_df_sources gs.cpp)
set(gs_hpx_grid_ro_sources gs.cpp)
set(gs_hpx_distributed_sources gs.cpp)

set(remote_functions_FLAGS DEPENDENCIES iostreams_component)
set(dataflow_test_FLAGS DEPENDENCIES iostreams_component bright_future_dataflow_component ${GOOGLE_PERFTOOLS_LIBRARY})
set(remote_object_FLAGS DEPENDENCIES remote_object_component iostreams_component)
set(vector_root_finder_FLAGS DEPENDENCIES iostreams_component)
set(gs_single_FLAGS DEPENDENCIES iostreams_component)
#set(gs_hpx_FLAGS DEPENDENCIES gs_hpx_component iostreams_component)
set(gs_hpx_blocked_FLAGS DEPENDENCIES gs_hpx_component iostreams_component)
set(gs_hpx_blocked_df_FLAGS DEPENDENCIES gs_hpx_component bright_future_dataflow_component iostreams_component ${GOOGLE_PERFTOOLS_LIBRARY})
set(gs_hpx_distributed_FLAGS DEPENDENCIES gs_hpx_component bright_future_dataflow_component iostreams_component ${GOOGLE_PERFTOOLS_LIBRARY})
set(gs_hpx_grid_ro_FLAGS DEPENDENCIES remote_object_component iostreams_component)

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
