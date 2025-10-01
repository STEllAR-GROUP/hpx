# Copyright (c) 2025 Alexandros Papadakis
# Copyright (c) 2025Panagiotis Syskakis
# HPX Contracts Module Unit Tests
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

if(HPX_WITH_TESTS)
    if (HPX_WITH_CONTRACTS)

        # Check if we have C++26 support for native contracts
        include(CheckCXXSourceCompiles)
        set(CMAKE_REQUIRED_FLAGS "-std=c++26")
        check_cxx_source_compiles("
            #if __cplusplus >= 202602L
            int main() { return 0; }
            #else
            #error \"C++26 not available\"
            #endif
        " HAS_CXX26)

        if(HAS_CXX26)
            # -------------------------------
            # C++26 Native Contract Tests
            # Condition: HPX_WITH_CONTRACTS=ON && __cplusplus >= 202602L
            # These tests use advanced C++26 contract features
            # Note: Global contract flags are set in root CMakeLists.txt
            # -------------------------------
            
            # Define contract types for native C++26 tests
            set(CONTRACT_TYPES "pre" "post" "cassert")

            # OBSERVE mode: Reports violations but does not abort
            foreach(CONTRACT_TYPE IN LISTS CONTRACT_TYPES)
                set(TARGET_NAME "contract_violation_${CONTRACT_TYPE}_observe")
                set(SOURCE_FILE "contract_violation_${CONTRACT_TYPE}.cpp")
                
                add_hpx_executable(
                    ${TARGET_NAME}_test INTERNAL_FLAGS
                    SOURCES ${SOURCE_FILE}
                    NOLIBS
                    DEPENDENCIES hpx_core
                    EXCLUDE_FROM_ALL
                    FOLDER "Tests/Unit/Modules/Core/Contracts/"
                )
                # Add only contract-specific evaluation semantic
                target_compile_options(${TARGET_NAME}_test PRIVATE
                    -fcontract-evaluation-semantic=observe
                )
                
                add_hpx_unit_test("modules.contracts" ${TARGET_NAME})
                set_tests_properties(tests.unit.modules.contracts.${TARGET_NAME} PROPERTIES
                    LABELS contracts
                )
            endforeach()

            # ENFORCE mode: Aborts on contract violation
            foreach(CONTRACT_TYPE IN LISTS CONTRACT_TYPES)
                set(TARGET_NAME "contract_violation_${CONTRACT_TYPE}_enforce")
                set(SOURCE_FILE "contract_violation_${CONTRACT_TYPE}.cpp")
                
                add_hpx_executable(
                    ${TARGET_NAME}_test INTERNAL_FLAGS
                    SOURCES ${SOURCE_FILE}
                    NOLIBS
                    DEPENDENCIES hpx_core
                    EXCLUDE_FROM_ALL
                    FOLDER "Tests/Unit/Modules/Core/Contracts/"
                )
                # Add only contract-specific evaluation semantic
                target_compile_options(${TARGET_NAME}_test PRIVATE
                    -fcontract-evaluation-semantic=enforce
                )
                
                add_hpx_unit_test("modules.contracts" ${TARGET_NAME})
                set_tests_properties(tests.unit.modules.contracts.${TARGET_NAME} PROPERTIES
                    WILL_FAIL TRUE
                    LABELS contracts
                )
            endforeach()

            # QUICK_ENFORCE mode: Aborts immediately on violation
            foreach(CONTRACT_TYPE IN LISTS CONTRACT_TYPES)
                set(TARGET_NAME "contract_violation_${CONTRACT_TYPE}_quick_enforce")
                set(SOURCE_FILE "contract_violation_${CONTRACT_TYPE}.cpp")
                
                add_hpx_executable(
                    ${TARGET_NAME}_test INTERNAL_FLAGS
                    SOURCES ${SOURCE_FILE}
                    NOLIBS
                    DEPENDENCIES hpx_core
                    EXCLUDE_FROM_ALL
                    FOLDER "Tests/Unit/Modules/Core/Contracts/"
                )
                # Add only contract-specific evaluation semantic
                target_compile_options(${TARGET_NAME}_test PRIVATE
                    -fcontract-evaluation-semantic=quick_enforce
                )
                
                add_hpx_unit_test("modules.contracts" ${TARGET_NAME})
                set_tests_properties(tests.unit.modules.contracts.${TARGET_NAME} PROPERTIES
                    WILL_FAIL TRUE
                    LABELS contracts
                )
            endforeach()

            # IGNORE mode: Contracts are completely ignored
            foreach(CONTRACT_TYPE IN LISTS CONTRACT_TYPES)
                set(TARGET_NAME "contract_violation_${CONTRACT_TYPE}_ignore")
                set(SOURCE_FILE "contract_violation_${CONTRACT_TYPE}.cpp")
                
                add_hpx_executable(
                    ${TARGET_NAME}_test INTERNAL_FLAGS
                    SOURCES ${SOURCE_FILE}
                    NOLIBS
                    DEPENDENCIES hpx_core
                    EXCLUDE_FROM_ALL
                    FOLDER "Tests/Unit/Modules/Core/Contracts/"
                )
                # Add only contract-specific evaluation semantic
                target_compile_options(${TARGET_NAME}_test PRIVATE
                    -fcontract-evaluation-semantic=ignore
                )
                
                add_hpx_unit_test("modules.contracts" ${TARGET_NAME})
                set_tests_properties(tests.unit.modules.contracts.${TARGET_NAME} PROPERTIES
                    LABELS contracts
                )
            endforeach()

        else()
            # -------------------------------
            # Fallback Mode Tests
            # Condition: HPX_WITH_CONTRACTS=ON && __cplusplus < 202602L
            # Tests the fallback mechanism from HPX_CONTRACT_ASSERT to HPX_ASSERT
            # Note: Global C++23 flags are set in root CMakeLists.txt
            # -------------------------------
            
            add_hpx_executable(
                fallback_contracts_test INTERNAL_FLAGS
                SOURCES fallback_contracts.cpp
                NOLIBS
                DEPENDENCIES hpx_core
                EXCLUDE_FROM_ALL
                FOLDER "Tests/Unit/Modules/Core/Contracts/"
            )
            # No additional compile options needed - using global settings
            
            add_hpx_unit_test("modules.contracts" fallback_contracts)
            set_tests_properties(tests.unit.modules.contracts.fallback_contracts PROPERTIES
                LABELS contracts
            )
            
        endif()

    else()
        # -------------------------------
        # Contracts Disabled Tests  
        # Condition: HPX_WITH_CONTRACTS=OFF
        # Tests that contract macros are no-ops when contracts are disabled
        # Note: Standard C++ flags are used (no special contract flags)
        # -------------------------------
        
        add_hpx_executable(
            disabled_contracts_test INTERNAL_FLAGS
            SOURCES fallback_contracts.cpp
            NOLIBS
            DEPENDENCIES hpx_core
            EXCLUDE_FROM_ALL
            FOLDER "Tests/Unit/Modules/Core/Contracts/"
        )
        # No additional compile options needed - using global settings
        
        add_hpx_unit_test("modules.contracts" disabled_contracts)
        set_tests_properties(tests.unit.modules.contracts.disabled_contracts PROPERTIES
            LABELS contracts
        )
        
    endif()
endif()
