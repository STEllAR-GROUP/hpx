# Copyright (c) 2025 Srinivas Yadav Singanaboina
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

find_package(Hwloc)
if(NOT Hwloc_FOUND)
  hpx_error(
    "Hwloc could not be found, please specify Hwloc_ROOT to point to the correct location"
  )
endif()
