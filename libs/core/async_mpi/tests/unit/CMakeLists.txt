# Copyright (c) 2019 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(tests mpi_ring_async_executor algorithm_transform_mpi)

set(mpi_ring_async_executor_PARAMETERS THREADS_PER_LOCALITY 4 LOCALITIES 2
                                       RUNWRAPPER mpi
)

set(algorithm_transform_mpi_PARAMETERS LOCALITIES 2 RUNWRAPPER mpi)
set(algorithm_transform_mpi_DEPENDENCIES hpx_execution_test_utilities)

foreach(test ${tests})

  set(sources ${test}.cpp)

  source_group("Source Files" FILES ${sources})

  # add example executable
  add_hpx_executable(
    ${test}_test INTERNAL_FLAGS
    SOURCES ${sources} ${${test}_FLAGS}
    EXCLUDE_FROM_ALL
    DEPENDENCIES Mpi::mpi ${${test}_DEPENDENCIES}
    FOLDER "Tests/Unit/Modules/Core/AsyncMPI"
  )

  add_hpx_unit_test("modules.async_mpi" ${test} ${${test}_PARAMETERS})

endforeach()
