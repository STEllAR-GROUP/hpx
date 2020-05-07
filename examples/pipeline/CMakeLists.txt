# Copyright (c) 2018 Thomas Heller
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

if(NOT HPX_WITH_DISTRIBUTED_RUNTIME)
  return()
endif()

set(pipeline emitter worker collector)

foreach(app ${pipeline})
  add_hpx_executable(
    ${app} INTERNAL_FLAGS
    SOURCES ${app}.cpp
    FOLDER "Examples/Pipeline"
  )
  add_hpx_example_target_dependencies("pipeline" ${app})
endforeach()
