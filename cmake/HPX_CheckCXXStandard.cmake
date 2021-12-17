# Copyright (c) 2021 ETH Zurich
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

if(HPX_WITH_CXX11)
  hpx_error(
    "HPX_WITH_CXX11 is deprecated and the minimum C++ standard required by HPX is 17. Configure HPXLocal with HPXLocal_WITH_CXX_STANDARD instead."
  )
elseif(HPX_WITH_CXX14)
  hpx_error(
    "HPX_WITH_CXX14 is deprecated and the minimum C++ standard required by HPX is 17. Configure HPXLocal with HPXLocal_WITH_CXX_STANDARD instead."
  )
elseif(HPX_WITH_CXX17)
  hpx_warn(
    "HPX_WITH_CXX17 is deprecated. Configure HPXLocal with HPXLocal_WITH_CXX_STANDARD=17 instead."
  )

  if(HPX_WITH_FETCH_HPXLOCAL)
    set(HPX_CXX_STANDARD 17)
  endif()
elseif(HPX_WITH_CXX20)
  hpx_warn(
    "HPX_WITH_CXX20 is deprecated. Configure HPXLocal with HPXLocal_WITH_CXX_STANDARD=20 instead."
  )

  if(HPX_WITH_FETCH_HPXLOCAL)
    set(HPX_CXX_STANDARD 20)
  endif()
endif()

set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_DEFAULT 98)
