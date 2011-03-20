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

set(group naming)

set(testcases
    gid_type
    resolver_client
    resolver_server
    )

foreach(testcase ${testcases})
  add_hpx_test(${type}.${group}.${testcase}
               SOURCES ${testcase}.cpp)

  # depend on the executable
  add_dependencies(tests.${type}.${group}
                   ${type}.${group}.${testcase}_test)
endforeach()

