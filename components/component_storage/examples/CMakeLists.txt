# Copyright (c) 2019 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

if(HPX_WITH_EXAMPLES)
  add_hpx_pseudo_target(examples.components.component_storage)
  add_hpx_pseudo_dependencies(
    examples.components examples.components.component_storage
  )
  if(HPX_WITH_TESTS AND HPX_WITH_TESTS_EXAMPLES)
    add_hpx_pseudo_target(tests.examples.components.component_storage)
    add_hpx_pseudo_dependencies(
      tests.examples.components tests.examples.components.component_storage
    )
  endif()
endif()
