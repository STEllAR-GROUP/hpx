# Copyright (c) 2019 Hartmut Kaiser
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(tests
    lexical_cast_abstract_test
    lexical_cast_arrays_test
    lexical_cast_cstring_wrapper_test
    lexical_cast_empty_input_test
    lexical_cast_filesystem_test
    lexical_cast_float_types_test
    lexical_cast_inf_nan_test
    lexical_cast_integral_types_test
    lexical_cast_loopback_test
    lexical_cast_noncopyable_test
    lexical_cast_pointers_test
    lexical_cast_stream_detection_test
    lexical_cast_stream_traits_test
    lexical_cast_test
    lexical_cast_try_lexical_convert
    lexical_cast_vc8_bug_test
)

foreach(test ${tests})
  set(sources
      ${test}.cpp)

  source_group("Source Files" FILES ${sources})

  # add example executable
  add_hpx_executable(${test}_test
    INTERNAL_FLAGS
    SOURCES ${sources}
    ${${test}_FLAGS}
    EXCLUDE_FROM_ALL
    NOLIBS
    DEPENDENCIES hpx_lexical_cast hpx_testing
      hpx_filesystem
    FOLDER "Tests/Unit/Modules/LexicalCast")

  add_hpx_unit_test("modules.lexical_cast" ${test} ${${test}_PARAMETERS})
  target_compile_definitions(${test}_test PRIVATE -DHPX_MODULE_STATIC_LINKING)

endforeach()
