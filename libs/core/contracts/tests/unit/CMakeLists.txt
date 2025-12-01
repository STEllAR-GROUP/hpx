# Copyright (c) 2025 Alexandros Papadakis
# HPX Contracts Module Unit Tests
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

# Declaration contracts tests (C++26 declaration syntax) These run when
# __cpp_contracts is available
set(contract_tests
    declaration_contracts_succeed
    declaration_contracts_fail_pre
    declaration_contracts_fail_post
    declaration_contracts_fail_contract_assert
    fallback_contracts_succeed
    fallback_contracts_fail
    disabled_contracts
)

foreach(test ${contract_tests})
  add_hpx_executable(
    ${test}_test INTERNAL_FLAGS
    SOURCES ${test}.cpp
    NOLIBS
    DEPENDENCIES hpx_core
    EXCLUDE_FROM_ALL
    FOLDER "Tests/Unit/Modules/Core/Contracts/"
  )
  add_hpx_unit_test("modules.contracts" ${test})
endforeach()

# Set failure expectation for fallback failure test Fallback only fails in Debug
# mode (HPX_ASSERT or HPX_CONTRACT_ASSERT behavior)
set_tests_properties(
  tests.unit.modules.contracts.declaration_contracts_fail_contract_assert
  PROPERTIES WILL_FAIL $<$<CONFIG:Debug>:ON>
)

if(HPX_HAVE_CXX26_CONTRACTS)
  # Set failure expectations for declaration contract failure tests Native
  # contracts should fail when violated
  set_tests_properties(
    tests.unit.modules.contracts.declaration_contracts_fail_pre
    PROPERTIES WILL_FAIL ON
  )
  set_tests_properties(
    tests.unit.modules.contracts.declaration_contracts_fail_post
    PROPERTIES WILL_FAIL ON
  )
else()
  # Set failure expectation for fallback failure test Fallback only fails in
  # Debug mode (HPX_ASSERT behavior)
  set_tests_properties(
    tests.unit.modules.contracts.fallback_contracts_fail
    PROPERTIES WILL_FAIL $<$<CONFIG:Debug>:ON>
  )
endif()
