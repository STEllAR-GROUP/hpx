# Copyright (c) 2025 Alexandros Papadakis
# HPX Contracts Module Unit Tests
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

if(HPX_WITH_TESTS)
    if (HPX_WITH_CONTRACTS)
        # -------------------------------
        # Contract Tests when HPX_WITH_CONTRACTS=ON
        # Automatically detect contract mode and build appropriate tests
        # -------------------------------
        
        # Detect if native C++26 contracts are supported
        # Check both __cpp_contracts feature macro (standard) and actual syntax support (experimental)
        add_hpx_config_test(
            HPX_HAVE_NATIVE_CONTRACTS
            SOURCE "
                #ifdef __cpp_contracts
                // Standard C++26 contracts support detected
                int main() { return 0; }
                #else
                #error No __cpp_contracts feature macro
                #endif
            "
            OR 
            SOURCE "
                // Test actual contract syntax support (for experimental implementations)
                int main() pre(true) {
                    contract_assert(true);
                    return 0;
                }
            "
        )
        
        # Validate HPX_HAVE_ASSERTS_AS_CONTRACT_ASSERTS configuration
        if(HPX_HAVE_ASSERTS_AS_CONTRACT_ASSERTS AND NOT HPX_HAVE_NATIVE_CONTRACTS)
            message(WARNING "HPX_HAVE_ASSERTS_AS_CONTRACT_ASSERTS is enabled but native C++26 contracts "
                           "are not supported by this compiler. This means HPX_ASSERT will be mapped to "
                           "HPX_CONTRACT_ASSERT, which in turn maps back to HPX_ASSERT, providing no "
                           "enhancement. To get true contract assertion benefits, use a compiler with "
                           "native contract support.")
        endif()
        
        if(HPX_HAVE_NATIVE_CONTRACTS)
            message(STATUS "HPX Contracts: Native C++26 contracts detected - building declaration tests")
            
            # Declaration contracts tests (C++26 declaration syntax)
            # These run when __cpp_contracts is available
            set(contract_tests 
                declaration_contracts_succeed
                declaration_contracts_fail_pre 
                declaration_contracts_fail_post 
                declaration_contracts_fail_contract_assert
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
            
            # Set failure expectations for declaration contract failure tests
            # Native contracts should fail when violated
            set_tests_properties(
                tests.unit.modules.contracts.declaration_contracts_fail_pre PROPERTIES WILL_FAIL true
            )
            set_tests_properties(
                tests.unit.modules.contracts.declaration_contracts_fail_post PROPERTIES WILL_FAIL true
            )
            set_tests_properties(
                tests.unit.modules.contracts.declaration_contracts_fail_contract_assert PROPERTIES WILL_FAIL
                                                            $<$<CONFIG:Debug>:true>
            )
            
        else()
            message(WARNING "HPX Contracts: Native C++26 contracts are not supported by this compiler. "
                           "Using fallback mode where HPX_PRE/HPX_POST become no-ops and "
                           "HPX_CONTRACT_ASSERT maps to HPX_ASSERT. "
                           "For full contract support, use a C++26-compliant compiler or "
                           "experimental builds with contract support.")
            message(STATUS "HPX Contracts: Fallback mode detected - building fallback tests")
            
            # Fallback contracts tests (simple syntax for fallback mode)
            # These run when __cpp_contracts is NOT available
            set(contract_tests 
                fallback_contracts_succeed
                fallback_contracts_fail
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
            
            # Set failure expectation for fallback failure test
            # Fallback only fails in Debug mode (HPX_ASSERT behavior)
            set_tests_properties(
                tests.unit.modules.contracts.fallback_contracts_fail PROPERTIES WILL_FAIL
                                                            $<$<CONFIG:Debug>:true>
            )
        endif()

    else()
        # -------------------------------
        # Contracts Disabled Tests  
        # Condition: HPX_WITH_CONTRACTS=OFF
        # Tests that contract macros are no-ops when contracts are disabled
        # -------------------------------
        
        message(STATUS "HPX Contracts: Disabled mode - building disabled tests")
        
        add_hpx_executable(
            disabled_contracts_test INTERNAL_FLAGS
            SOURCES disabled_contracts.cpp
            NOLIBS
            DEPENDENCIES hpx_core
            EXCLUDE_FROM_ALL
            FOLDER "Tests/Unit/Modules/Core/Contracts/"
        )
        
        add_hpx_unit_test("modules.contracts" disabled_contracts)
        
    endif()
endif()
