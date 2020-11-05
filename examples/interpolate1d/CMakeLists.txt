# Copyright (c) 2007-2017 Hartmut Kaiser
# Copyright (c) 2011      Bryce Lelbach
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

if(HPX_WITH_EXAMPLES_HDF5)
  if(NOT HDF5_CXX_LIBRARIES OR NOT HDF5_IS_PARALLEL)
    hpx_error(
      "The Interpolate1D example requires HDF C++ library enabled for thread-safety"
    )
  endif()

  # just recurse into all components subdirectories and execute the scripts
  # there
  set(subdirs interpolate1d)

  set(interpolate1d_PARAMETERS THREADS_PER_LOCALITY 4)

  # for all targets specified above
  foreach(subdir ${subdirs})
    add_subdirectory(${subdir})

    set(sources ${subdir}_client.cpp)

    source_group("Source Files" FILES ${sources})

    # add example executable
    add_hpx_executable(
      ${subdir}_client INTERNAL_FLAGS
      SOURCES ${sources}
      DEPENDENCIES ${subdir}_component
      FOLDER "Examples/Interpolate1D"
    )

    add_hpx_example_target_dependencies("interpolate1d" ${subdir}_client)

    if(HPX_WITH_TESTS AND HPX_WITH_TESTS_EXAMPLES)
      add_hpx_example_test(
        "interpolate1d" ${subdir}_client ${${subdir}_PARAMETERS}
      )
    endif()
  endforeach()

  add_hpx_executable(
    create_1d_testdata INTERNAL_FLAGS
    SOURCES create_testdata.cpp
    NOLIBS
    FOLDER "Examples/Interpolate1D"
  )

  target_include_directories(create_1d_testdata PRIVATE ${HDF5_INCLUDE_DIRS})
  target_link_libraries(create_1d_testdata PRIVATE ${HDF5_CXX_LIBRARIES})
  add_hpx_example_target_dependencies("interpolate1d" create_1d_testdata)

endif()
