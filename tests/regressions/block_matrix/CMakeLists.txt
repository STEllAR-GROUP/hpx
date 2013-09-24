# Copyright (c) 2013 Erik Schnetter
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

if(HPX_HAVE_CXX11)

  set(tests block_matrix)

  set(sources
      algorithms.cc block_matrix.cc main.cc matrix.cc matrix_hpx.cc tests.cc)

  source_group("Source Files" FILES ${sources})

  add_definitions(-DHPX_ACTION_ARGUMENT_LIMIT=6)

  add_hpx_executable(block_matrix_test
                     SOURCES ${sources}
                     ${block_matrix_FLAGS}
                     FOLDER "Tests/Regressions/Block-Matrix")

  add_hpx_regression_test("block_matrix" block_matrix ${block_matrix_PARAMETERS})

  add_hpx_pseudo_target(tests.regressions.block_matrix.block_matrix)

  add_hpx_pseudo_dependencies(tests.regressions.block_matrix
                              tests.regressions.block_matrix.block_matrix)

  add_hpx_pseudo_dependencies(tests.regressions.block_matrix.block_matrix
                              block_matrix_test_exe)

endif()
