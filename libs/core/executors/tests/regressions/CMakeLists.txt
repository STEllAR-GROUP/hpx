# Copyright (c) 2021 ETH Zurich
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

if(HPX_WITH_COMPILE_ONLY_TESTS)
  set(compile_tests)
  if(HPX_WITH_CUDA OR HPX_WITH_HIP)
    list(APPEND compile_tests service_executor_cuda)
    set(service_executor_cuda_CUDA ON)
  endif()

  foreach(compile_test ${compile_tests})
    if(${${compile_test}_CUDA})
      set(sources ${compile_test}.cu)
    else()
      set(sources ${compile_test}.cpp)
    endif()

    source_group("Source Files" FILES ${sources})

    add_hpx_regression_compile_test(
      "modules.executors" ${compile_test}
      SOURCES ${sources} ${${compile_test}_FLAGS}
      FOLDER "Tests/Regressions/Modules/Core/Executors/CompileOnly"
    )
  endforeach()
endif()
