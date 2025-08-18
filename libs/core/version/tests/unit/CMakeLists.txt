# Copyright (c) 2025 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(tests version_api)

foreach(test ${tests})
  set(sources ${test}.cpp)

  source_group("Source Files" FILES ${sources})

  set(folder_name "Tests/Unit/Modules/Core/Version")

  add_hpx_executable(
    ${test}_test INTERNAL_FLAGS
    SOURCES ${sources} ${${test}_FLAGS}
    EXCLUDE_FROM_ALL
    FOLDER ${folder_name}
  )

  if(HPX_WITH_CXX_MODULES)
    if(TARGET hpx_core)
      target_link_libraries(${test}_test PRIVATE hpx_core)
    endif()
    set_property(TARGET ${test}_test PROPERTY CXX_SCAN_FOR_MODULES ON)
  endif()

  add_hpx_unit_test("modules.version" ${test} ${${test}_PARAMETERS})
endforeach()
