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

set(group unit.boost.lockfree.deque) 

set(testcases
    lifo
    fifo
    deque    
    bit_manipulation)

foreach(testcase ${testcases})
  add_hpx_test(${group}.${testcase} SOURCES ${testcase}.cpp)

  # depend on the executable
  add_hpx_pseudo_dependencies(${group} ${group}.${testcase}_test)
endforeach()

