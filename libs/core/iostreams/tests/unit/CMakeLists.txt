# Copyright (c) 2020-2026 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(tests
    array
    auto_close
    bool_traits
    buffer_size
    closable
    combine_streams
    component_access
    compose_streams
    copy_streams
    direct_adapter
    dual_seekable
    execute_all
    file
    filtering_stream
    filters
    flush_streams
    null_streams
    pipeline_streams
    read_nonblocking
    seekable_file
    seekable_filter
    sequence_streams
    stream_state
    symmetric_filter
    write_failure
)

# Add the tests
foreach(test ${tests})
  set(sources ${test}.cpp)

  source_group("Source Files" FILES ${sources})

  add_hpx_executable(
    ${test}_test INTERNAL_FLAGS
    SOURCES ${sources} ${${test}_FLAGS} ${${test}_LIBS}
    EXCLUDE_FROM_ALL
    HPX_PREFIX ${HPX_BUILD_PREFIX}
    FOLDER "Tests/Unit/Modules/Core/IOStreams"
  )

  add_hpx_unit_test("modules.iostreams" ${test} ${${test}_PARAMETERS})
endforeach()
