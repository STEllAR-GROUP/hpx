# Copyright (c) 2014-2019 Hartmut Kaiser
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(subdirs inspect)

foreach(subdir ${subdirs})
  add_hpx_pseudo_target(tools.${subdir})
  add_subdirectory(${subdir})
  add_hpx_pseudo_dependencies(tools tools.${subdir})
endforeach()
