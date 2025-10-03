# Copyright (c) 2025 Alexandros Papadakis
# Copyright (c) 2025Panagiotis Syskakis
# HPX Contracts Module Unit Tests
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

if(HPX_WITH_TESTS)
    if (HPX_WITH_CONTRACTS)
        # -------------------------------
        # Contract Tests when HPX_WITH_CONTRACTS=ON
        # This includes both native (__cpp_contracts) and fallback modes
        # We need runtime detection to run appropriate tests
        # -------------------------------
        
        # Declaration contracts tests (C++26 declaration syntax)
        # These should work when __cpp_contracts is available
        set(declaration_tests 
            declaration_contracts_succeed
            declaration_contracts_fail_pre 
            declaration_contracts_fail_post 
            declaration_contracts_fail_contract_assert
        )
        foreach(test ${declaration_tests})
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
        
        # Set failure expectations for declaration contract failure tests
        set_tests_properties(
            tests.unit.modules.contracts.declaration_contracts_fail_pre PROPERTIES WILL_FAIL
                                                        
        )
        set_tests_properties(
            tests.unit.modules.contracts.declaration_contracts_fail_post PROPERTIES WILL_FAIL
                                                        
        )
        set_tests_properties(
            tests.unit.modules.contracts.declaration_contracts_fail_contract_assert PROPERTIES WILL_FAIL
                                                        
        )
        
        # Fallback contracts tests (simple syntax for fallback mode)
        # These work when __cpp_contracts is NOT available
        set(fallback_tests 
            fallback_contracts_succeed
            fallback_contracts_fail
        )
        foreach(test ${fallback_tests})
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
        
        # Set failure expectation for fallback failure test
        set_tests_properties(
            tests.unit.modules.contracts.fallback_contracts_fail PROPERTIES WILL_FAIL
                                                        $<$<CONFIG:Debug>:true>
        )

    else()
        # -------------------------------
        # Contracts Disabled Tests  
        # Condition: HPX_WITH_CONTRACTS=OFF
        # Tests that contract macros are no-ops when contracts are disabled
        # -------------------------------
        
        add_hpx_executable(
            disabled_contracts_test INTERNAL_FLAGS
            SOURCES disabled_contracts.cpp
            NOLIBS
            DEPENDENCIES hpx_core
            EXCLUDE_FROM_ALL
            FOLDER "Tests/Unit/Modules/Core/Contracts/"
        )
        
        add_hpx_unit_test("modules.contracts" disabled_contracts)
        set_tests_properties(tests.unit.modules.contracts.disabled_contracts PROPERTIES
            LABELS contracts
        )
        
    endif()
endif()
