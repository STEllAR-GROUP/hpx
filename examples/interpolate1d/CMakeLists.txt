# Copyright (c) 2007-2017 Hartmut Kaiser
# Copyright (c) 2011      Bryce Lelbach
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

if(HPX_WITH_EXAMPLES_HDF5)
  if(NOT HDF5_CXX_LIBRARIES OR NOT HDF5_IS_PARALLEL)
    hpx_error("The Interpolate1D example requires HDF C++ library enabled for thread-safety")
  endif()

  include_directories(${HDF5_INCLUDE_DIRS})

  # just recurse into all components subdirectories and execute the scripts there
  set(subdirs
      interpolate1d)

  # for all targets specified above
  foreach(subdir ${subdirs})
    add_subdirectory(${subdir})

    set(sources
        ${subdir}_client.cpp)

    source_group("Source Files" FILES ${sources})

    # add example executable
    add_hpx_executable(${subdir}_client
      SOURCES ${sources}
      DEPENDENCIES ${subdir}_component
      FOLDER "Examples/Interpolate1D")

    # add a custom target for this example
    add_hpx_pseudo_target(examples.interpolate1d.${subdir}_)

    # make pseudo-targets depend on master pseudo-target
    add_hpx_pseudo_dependencies(examples.interpolate1d
                                examples.interpolate1d.${subdir}_)

    # add dependencies to pseudo-target
    add_hpx_pseudo_dependencies(examples.interpolate1d.${subdir}_
                                ${subdir}_client_exe)
  endforeach()

  add_hpx_executable(create_1d_testdata
    SOURCES create_testdata.cpp
    DEPENDENCIES ${HDF5_LIBRARIES}
    NOLIBS
    FOLDER "Examples/Interpolate1D")

  # add a custom target for this example
  add_hpx_pseudo_target(examples.interpolate1d.create_1d_testdata)

  # make pseudo-targets depend on master pseudo-target
  add_hpx_pseudo_dependencies(examples.interpolate1d
                              examples.interpolate1d.create_1d_testdata)

  # add dependencies to pseudo-target
  add_hpx_pseudo_dependencies(examples.interpolate1d.create_1d_testdata
                              create_1d_testdata_exe)

endif()
