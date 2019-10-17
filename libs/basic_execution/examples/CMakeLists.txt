# Copyright (c) 2019 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

if (HPX_WITH_EXAMPLES)
  add_hpx_pseudo_target(examples.modules.basic_execution)
  add_hpx_pseudo_dependencies(examples.modules examples.modules.basic_execution)
  if (HPX_WITH_TESTS AND HPX_WITH_TESTS_EXAMPLES AND HPX_BASIC_EXECUTION_WITH_TESTS)
    add_hpx_pseudo_target(tests.examples.modules.basic_execution)
    add_hpx_pseudo_dependencies(tests.examples.modules tests.examples.modules.basic_execution)
  endif()

  find_package(MPI)

  if(NOT MPI_CXX_FOUND)
    hpx_error("MPI could not be found and HPX_WITH_PARCELPORT_MPI=On, please specify MPI_CXX_COMPILER to point to a working MPI C++ compiler for your platform")
  endif()

  add_executable(mpi_ping_pong ping_pong.cpp)
  hpx_setup_target(mpi_ping_pong)
  target_link_libraries(mpi_ping_pong PRIVATE hpx_functional ${MPI_CXX_LIB_NAMES} )
  target_include_directories(mpi_ping_pong PRIVATE ${MPI_CXX_INCLUDE_PATH})
  target_compile_options(mpi_ping_pong PRIVATE -fopenmp)
  target_link_options(mpi_ping_pong PRIVATE -fopenmp)
endif()
