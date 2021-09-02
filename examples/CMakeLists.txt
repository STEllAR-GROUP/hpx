# Copyright (c) 2007-2017 Hartmut Kaiser
# Copyright (c) 2011      Bryce Lelbach
# Copyright (c) 2011      Dylan Stark
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(subdirs 1d_stencil balancing future_reduce jacobi_smp quickstart transpose)

foreach(subdir ${subdirs})
  if(HPXLocal_WITH_TESTS AND HPXLocal_WITH_TESTS_EXAMPLES)
    hpx_local_add_pseudo_target(tests.examples.${subdir})
    hpx_local_add_pseudo_dependencies(tests.examples tests.examples.${subdir})
  endif()

  hpx_local_add_pseudo_target(examples.${subdir})
  add_subdirectory(${subdir})
  hpx_local_add_pseudo_dependencies(examples examples.${subdir})
endforeach()
