# Copyright (c) 2011 Thomas Heller
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

add_definitions(-DHPX_ACTION_ARGUMENT_LIMIT=10)
add_definitions(-DHPX_COMPONENT_CREATE_ARGUMENT_LIMIT=10)
add_definitions(-DHPX_FUNCTION_LIMIT=15)

set(bright_future_programs
        root_finder
        vector_root_finder
   )

if(HPX_CXX11_LAMBDAS)
  set(bright_future_programs ${bright_future_programs}
        gs_hpx_distributed
        dataflow_overhead
        dataflow_overhead2
        gs_single
        gs_hpx_blocked
        gs_hpx_grid_ro
        remote_functions
        gs_hpx_blocked_df
        jacobi_hpx
        jacobi_hpx_simple
        jacobi_hpx_distributed_simple
        jacobi_hpx_distributed_local_deps
        jacobi_hpx_distributed_simple_2
        jacobi_nonuniform_hpx
      )

  if(OPENMP_FOUND)
    set(bright_future_programs ${bright_future_programs}
      jacobi_openmp
      jacobi_openmp_simple
      jacobi_openmp_dynamic_simple
      jacobi_nonuniform_openmp
      jacobi_nonuniform_openmp_dynamic
    )
  endif()

  if(MPI_CXX_FOUND AND OPENMP_FOUND)
    set(bright_future_programs ${bright_future_programs}
      jacobi_mpi_simple
    )
  endif()

  add_hpx_component(
    gs_hpx
    SOURCES server/remote_lse.cpp
    HEADERS server/remote_lse.hpp grid.hpp
    DEPENDENCIES iostreams_component distributing_factory_component
    FOLDER "Examples/BrightFuture"
  )
endif()

set(gs_single_sources gs.cpp grid.cpp)
set(jacobi_hpx_sources gs.cpp grid.cpp)
set(jacobi_hpx_simple_sources gs.cpp grid.cpp)
set(jacobi_hpx_distributed_simple_sources gs.cpp grid.cpp)
set(jacobi_hpx_distributed_simple_2_sources gs.cpp grid.cpp)
set(jacobi_nonuniform_hpx_sources jacobi_nonuniform.cpp grid.cpp)
set(gs_hpx_blocked_sources gs.cpp grid.cpp)
set(gs_hpx_blocked_df_sources gs.cpp grid.cpp)
set(gs_hpx_grid_ro_sources gs.cpp grid.cpp)
set(gs_hpx_distributed_sources gs.cpp grid.cpp)

if(OPENMP_FOUND)
  set(jacobi_openmp_sources gs.cpp)
  set(jacobi_openmp_simple_sources gs.cpp grid.cpp)
  set(jacobi_openmp_dynamic_simple_sources gs.cpp grid.cpp)
  set(jacobi_nonuniform_openmp_sources jacobi_nonuniform.cpp grid.cpp)
  set(jacobi_nonuniform_openmp_dynamic_sources jacobi_nonuniform.cpp grid.cpp)
endif()

if(MPI_CXX_FOUND AND OPENMP_FOUND)
  set(jacobi_mpi_simple_sources gs.cpp)
endif()

set(remote_functions_FLAGS DEPENDENCIES iostreams_component)
set(vector_root_finder_FLAGS DEPENDENCIES iostreams_component)
if(HPX_CXX11_LAMBDAS)
  set(gs_hpx_blocked_FLAGS DEPENDENCIES gs_hpx_component iostreams_component)
  set(gs_hpx_blocked_df_FLAGS DEPENDENCIES gs_hpx_component dataflow_component iostreams_component)
  set(gs_hpx_distributed_FLAGS DEPENDENCIES gs_hpx_component dataflow_component iostreams_component)
  set(dataflow_overhead_FLAGS DEPENDENCIES dataflow_component iostreams_component)
  set(dataflow_overhead2_FLAGS DEPENDENCIES dataflow_component iostreams_component)
  set(jacobi_hpx_FLAGS DEPENDENCIES dataflow_component iostreams_component remote_object_component)
  set(jacobi_hpx_simple_FLAGS DEPENDENCIES dataflow_component iostreams_component remote_object_component)
  set(jacobi_hpx_distributed_simple_FLAGS DEPENDENCIES dataflow_component iostreams_component remote_object_component)
  set(jacobi_hpx_distributed_simple_2_FLAGS DEPENDENCIES dataflow_component iostreams_component remote_object_component)
  set(jacobi_nonuniform_hpx_FLAGS DEPENDENCIES dataflow_component iostreams_component remote_object_component)
  set(gs_hpx_grid_ro_FLAGS DEPENDENCIES remote_object_component iostreams_component)
endif()

foreach(bright_future_program ${bright_future_programs})
  set(sources
       ${${bright_future_program}_sources} ${bright_future_program}.cpp)

  source_group("Source Files" FILES ${sources})

  # add bright_future executable
  add_hpx_executable(${bright_future_program}
                     MODULE bright_future
                     SOURCES ${sources}
                     ${${bright_future_program}_FLAGS}
                     FOLDER "Examples/BrightFuture")

  # add a custom target for this example
  add_hpx_pseudo_target(examples.bright_future.${bright_future_program})

  # make pseudo-targets depend on master pseudo-target
  add_hpx_pseudo_dependencies(examples.bright_future
                              examples.bright_future.${bright_future_program})

  # add dependencies to pseudo-target
  add_hpx_pseudo_dependencies(examples.bright_future.${bright_future_program}
                              ${bright_future_program}_exe)
endforeach()

if(HPX_CXX11_LAMBDAS)
  set_target_properties(gs_single_exe PROPERTIES COMPILE_DEFINITIONS BRIGHT_FUTURE_NO_HPX)

  if(OPENMP_FOUND)
    set_target_properties(gs_single_exe PROPERTIES COMPILE_FLAGS ${OpenMP_CXX_FLAGS})
    set_target_properties(gs_single_exe PROPERTIES LINK_FLAGS "${OpenMP_CXX_FLAGS}")

    set_target_properties(jacobi_openmp_exe PROPERTIES COMPILE_DEFINITIONS BRIGHT_FUTURE_NO_HPX)
    set_target_properties(jacobi_openmp_exe PROPERTIES COMPILE_FLAGS "${OpenMP_CXX_FLAGS} -DOPENMP_GRID")
    set_target_properties(jacobi_openmp_exe PROPERTIES LINK_FLAGS "${OpenMP_CXX_FLAGS}")

    set_target_properties(jacobi_openmp_simple_exe PROPERTIES COMPILE_DEFINITIONS BRIGHT_FUTURE_NO_HPX)
    set_target_properties(jacobi_openmp_simple_exe PROPERTIES COMPILE_FLAGS "${OpenMP_CXX_FLAGS} -DOPENMP_GRID")
    set_target_properties(jacobi_openmp_simple_exe PROPERTIES LINK_FLAGS "${OpenMP_CXX_FLAGS}")

    set_target_properties(jacobi_openmp_dynamic_simple_exe PROPERTIES COMPILE_DEFINITIONS BRIGHT_FUTURE_NO_HPX)
    set_target_properties(jacobi_openmp_dynamic_simple_exe PROPERTIES COMPILE_FLAGS "${OpenMP_CXX_FLAGS} -DOPENMP_GRID")
    set_target_properties(jacobi_openmp_dynamic_simple_exe PROPERTIES LINK_FLAGS "${OpenMP_CXX_FLAGS}")

    set_target_properties(jacobi_nonuniform_openmp_exe PROPERTIES COMPILE_DEFINITIONS BRIGHT_FUTURE_NO_HPX)
    set_target_properties(jacobi_nonuniform_openmp_exe PROPERTIES COMPILE_FLAGS "${OpenMP_CXX_FLAGS} -DOPENMP_GRID")
    set_target_properties(jacobi_nonuniform_openmp_exe PROPERTIES LINK_FLAGS "${OpenMP_CXX_FLAGS}")
    
    set_target_properties(jacobi_nonuniform_openmp_dynamic_exe PROPERTIES COMPILE_DEFINITIONS BRIGHT_FUTURE_NO_HPX)
    set_target_properties(jacobi_nonuniform_openmp_dynamic_exe PROPERTIES COMPILE_FLAGS "${OpenMP_CXX_FLAGS} -DOPENMP_GRID")
    set_target_properties(jacobi_nonuniform_openmp_dynamic_exe PROPERTIES LINK_FLAGS "${OpenMP_CXX_FLAGS}")
  endif()

  if(MPI_CXX_FOUND AND OPENMP_FOUND)
    include_directories(SYSTEM ${MPI_CXX_INCLUDE_PATH})
    target_link_libraries(jacobi_mpi_simple_exe ${MPI_CXX_LIBRARIES})
    set_target_properties(jacobi_mpi_simple_exe PROPERTIES COMPILE_DEFINITIONS BRIGHT_FUTURE_NO_HPX)
    set_target_properties(jacobi_mpi_simple_exe PROPERTIES COMPILE_FLAGS "${MPI_CXX_COMPILE_FLAGS} ${OpenMP_CXX_FLAGS} -DOPENMP_GRID -DBRIGHT_FUTURE_WITH_MPI")
    set_target_properties(jacobi_mpi_simple_exe PROPERTIES LINK_FLAGS "${MPI_CXX_LINK_FLAGS} ${OpenMP_CXX_FLAGS}")
  endif()
endif()
