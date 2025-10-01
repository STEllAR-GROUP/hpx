# This CMakeLists.txt configures tests for the HPX contracts module.
#currently a copy of the one in assertions



include(HPX_Message)
include(HPX_Option)

if(HPX_WITH_TESTS)
    if(HPX_WITH_TESTS_UNIT)
        add_hpx_pseudo_target(tests.unit.modules.contracts)
        add_hpx_pseudo_dependencies(tests.unit.modules tests.unit.modules.contracts)
        add_subdirectory(unit)
    endif()

    if(HPX_WITH_TESTS_REGRESSIONS)
        add_hpx_pseudo_target(tests.regressions.modules.contracts)
        add_hpx_pseudo_dependencies(
            tests.regressions.modules tests.regressions.modules.contracts
        )
        add_subdirectory(regressions)
    endif()

    if(HPX_WITH_TESTS_BENCHMARKS)
        add_hpx_pseudo_target(tests.performance.modules.contracts)
        add_hpx_pseudo_dependencies(
            tests.performance.modules tests.performance.modules.contracts
        )
        add_subdirectory(performance)
    endif()

    if(HPX_WITH_TESTS_HEADERS)
        add_hpx_header_tests(
            modules.contracts
            HEADERS ${contracts_headers}
            HEADER_ROOT ${PROJECT_SOURCE_DIR}/include
            NOLIBS
            DEPENDENCIES hpx_contracts
        )
    endif()
endif()