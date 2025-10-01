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
            # -------------------------------
            
            # Define contract types for native C++26 tests
            set(CONTRACT_TYPES "pre" "post" "cassert")

            # OBSERVE mode: Reports violations but does not abort
            foreach(CONTRACT_TYPE IN LISTS CONTRACT_TYPES)
                set(TARGET_NAME "contract_violation_${CONTRACT_TYPE}_observe")
                set(SOURCE_FILE "contract_violation_${CONTRACT_TYPE}.cpp")
                
                add_executable(${TARGET_NAME} ${SOURCE_FILE})
                target_compile_options(${TARGET_NAME} PRIVATE
                    -std=c++26 -fcontracts -stdlib=libc++
                    -fcontract-evaluation-semantic=observe
                )
                target_link_libraries(${TARGET_NAME} PRIVATE hpx_core)
                
                add_hpx_unit_test("contracts" ${TARGET_NAME})
                set_tests_properties(${TARGET_NAME} PROPERTIES
                    LABELS contracts
                )
            endforeach()

            # ENFORCE mode: Aborts on contract violation
            foreach(CONTRACT_TYPE IN LISTS CONTRACT_TYPES)
                set(TARGET_NAME "contract_violation_${CONTRACT_TYPE}_enforce")
                set(SOURCE_FILE "contract_violation_${CONTRACT_TYPE}.cpp")
                
                add_executable(${TARGET_NAME} ${SOURCE_FILE})
                target_compile_options(${TARGET_NAME} PRIVATE
                    -std=c++26 -fcontracts -stdlib=libc++
                    -fcontract-evaluation-semantic=enforce
                )
                target_link_libraries(${TARGET_NAME} PRIVATE hpx_core)
                
                add_hpx_unit_test("contracts" ${TARGET_NAME})
                set_tests_properties(${TARGET_NAME} PROPERTIES
                    WILL_FAIL TRUE
                    LABELS contracts
                )
            endforeach()

            # QUICK_ENFORCE mode: Aborts immediately on violation
            foreach(CONTRACT_TYPE IN LISTS CONTRACT_TYPES)
                set(TARGET_NAME "contract_violation_${CONTRACT_TYPE}_quick_enforce")
                set(SOURCE_FILE "contract_violation_${CONTRACT_TYPE}.cpp")
                
                add_executable(${TARGET_NAME} ${SOURCE_FILE})
                target_compile_options(${TARGET_NAME} PRIVATE
                    -std=c++26 -fcontracts -stdlib=libc++
                    -fcontract-evaluation-semantic=quick_enforce
                )
                target_link_libraries(${TARGET_NAME} PRIVATE hpx_core)
                
                add_hpx_unit_test("contracts" ${TARGET_NAME})
                set_tests_properties(${TARGET_NAME} PROPERTIES
                    WILL_FAIL TRUE
                    LABELS contracts
                )
            endforeach()

            # IGNORE mode: Contracts are completely ignored
            foreach(CONTRACT_TYPE IN LISTS CONTRACT_TYPES)
                set(TARGET_NAME "contract_violation_${CONTRACT_TYPE}_ignore")
                set(SOURCE_FILE "contract_violation_${CONTRACT_TYPE}.cpp")
                
                add_executable(${TARGET_NAME} ${SOURCE_FILE})
                target_compile_options(${TARGET_NAME} PRIVATE
                    -std=c++26 -fcontracts -stdlib=libc++
                    -fcontract-evaluation-semantic=ignore
                )
                target_link_libraries(${TARGET_NAME} PRIVATE hpx_core)
                
                add_hpx_unit_test("contracts" ${TARGET_NAME})
                set_tests_properties(${TARGET_NAME} PROPERTIES
                    LABELS contracts
                )
            endforeach()

        else()
            # -------------------------------
            # Fallback Mode Tests
            # Condition: HPX_WITH_CONTRACTS=ON && __cplusplus < 202602L
            # Tests the fallback mechanism from HPX_CONTRACT_ASSERT to HPX_ASSERT
            # -------------------------------
            
            add_executable(fallback_contracts fallback_contracts.cpp)
            target_compile_options(fallback_contracts PRIVATE
                -std=c++23  # Deliberately use C++23 to test fallback
            )
            target_link_libraries(fallback_contracts PRIVATE hpx_core)
            
            add_hpx_unit_test("contracts" fallback_contracts)
            set_tests_properties(fallback_contracts PROPERTIES
                LABELS contracts
            )
            
        endif()

    else()
        # -------------------------------
        # Contracts Disabled Tests  
        # Condition: HPX_WITH_CONTRACTS=OFF
        # Tests that contract macros are no-ops when contracts are disabled
        # -------------------------------
        
        add_executable(disabled_contracts fallback_contracts.cpp)
        target_compile_options(disabled_contracts PRIVATE
            -std=c++23  # Any standard works when contracts are disabled
        )
        target_link_libraries(disabled_contracts PRIVATE hpx_core)
        
        add_hpx_unit_test("contracts" disabled_contracts)
        set_tests_properties(disabled_contracts PROPERTIES
            LABELS contracts
        )
        
    endif()
endif()
