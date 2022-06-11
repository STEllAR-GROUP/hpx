# Copyright (c) 2020-2021 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)


set(tests sycl_vector_add)

foreach (test ${tests})
    hpx_info("Building SYCL test ${test}_test")
    set(sources "${test}.cpp" "${additional_sources}")

    add_hpx_executable(
        ${test}_test INTERNAL_FLAGS
        SOURCES ${sources} ${${test}_FLAGS}
        DEPENDENCIES ${${test_program}_FLAGS}
        EXCLUDE_FROM_ALL
        HPX_PREFIX ${HPX_BUILD_PREFIX}
        FOLDER "Tests/Unit/Modules/Core/AsyncSYCL"
        )

      # make it compile with sycl
      target_compile_options(${test}_test PUBLIC -fsycl)
      # Add to tests
      add_hpx_unit_test(
           "modules.async_sycl" ${test} ${${test}_PARAMETERS} RUN_SERIAL
           )
endforeach()
