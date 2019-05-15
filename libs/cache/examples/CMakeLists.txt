# Copyright (c) 2019 The STE||AR-Group
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

if (HPX_WITH_TESTS_EXAMPLES)
  add_hpx_pseudo_target(tests.examples.cache)
  add_hpx_pseudo_dependencies(tests.examples tests.examples.cache)
endif()

