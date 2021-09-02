# Copyright (c) 2020 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

if(HPXLocal_WITH_EXAMPLES)
  hpx_local_add_pseudo_target(examples.modules.properties)
  hpx_local_add_pseudo_dependencies(
    examples.modules examples.modules.properties
  )
  if(HPXLocal_WITH_TESTS AND HPXLocal_WITH_TESTS_EXAMPLES)
    hpx_local_add_pseudo_target(tests.examples.modules.properties)
    hpx_local_add_pseudo_dependencies(
      tests.examples.modules tests.examples.modules.properties
    )
  endif()
endif()
