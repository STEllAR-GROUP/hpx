################################################################################
# Copyright (c) 2003 & onward LASMEA UMR 6602 CNRS/Univ. Clermont II
# Copyright (c) 2009 & onward LRI    UMR 8623 CNRS/Univ Paris Sud XI
# Copyright (c) 2011 Bryce Lelbach 
#
# Derived from the NT2 CMake unit tests (http://nt2.metascale.org/)
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying 
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
################################################################################

set(group unit.hpx.runtime.agas.tcpip_stdmap)

set(testcases
    legacy_agent_serial)

foreach(testcase ${testcases})
  add_hpx_test(${group}.${testcase}
               SOURCES ${testcase}.cpp
               DEPENDENCIES agas_stdmap_component
                            agas_stdmap_tcpip_component
               ARGS "-r" "--agas" "${HPX_TEST_AGAS_ADDRESS}"
                         "--hpx" "${HPX_TEST_PARCEL_ADDRESS}"
                         "-t" "${HPX_TEST_NUM_THREADS}"
                         "-I\"system.prefix=${CMAKE_BINARY_DIR}\"")

  # depend on the executable
  add_hpx_pseudo_dependencies(${group} ${group}.${testcase}_test)
endforeach()

set(subdirs 
    component_namespace
    primary_namespace
    symbol_namespace)

foreach(subdir ${subdirs})
  add_hpx_pseudo_target(${group}.${subdir})
  add_subdirectory(${subdir})
  add_hpx_pseudo_dependencies(${group} ${group}.${subdir})
endforeach()

