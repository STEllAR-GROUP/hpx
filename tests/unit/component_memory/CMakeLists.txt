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

set(type unit)

set(group component_memory)

set(testcases
    basic 
    )

foreach(testcase ${testcases})
  add_hpx_test(${group}_${testcase}
               ${type}.${group}.${testcase}
               SOURCES ${group}_${testcase}.cpp
               ARGS "-r")

  # add a custom target for this test
  add_custom_target(tests.${type}.${group}.${testcase})

  # make parent depend on us
  add_dependencies(tests.${type}.${group}
                   tests.${type}.${group}.${testcase})

  # depend on the executable
  add_dependencies(tests.${type}.${group}.${testcase}
                   ${group}_${testcase}_test)
endforeach()

