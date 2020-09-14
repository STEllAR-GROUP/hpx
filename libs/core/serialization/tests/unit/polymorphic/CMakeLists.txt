# Copyright (c) 2014 Bryce Adelstein-Lelbach
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(tests
    polymorphic_reference
    polymorphic_pointer
    polymorphic_nonintrusive
    polymorphic_nonintrusive_abstract
    polymorphic_semiintrusive_template
    polymorphic_template
    smart_ptr_polymorphic
    smart_ptr_polymorphic_nonintrusive
)

foreach(test ${tests})
  set(sources ${test}.cpp)

  source_group("Source Files" FILES ${sources})

  # add example executable
  add_hpx_executable(
    ${test}_test INTERNAL_FLAGS
    SOURCES ${sources} ${${test}_FLAGS}
    EXCLUDE_FROM_ALL NOLIBS
    DEPENDENCIES hpx_serialization hpx_testing
    FOLDER "Tests/Unit/Modules/Core/Serialization/Polymorphic"
  )

  add_hpx_unit_test("modules.serialization" ${test} ${${test}_PARAMETERS})
  target_compile_definitions(${test}_test PRIVATE -DHPX_MODULE_STATIC_LINKING)

endforeach()
