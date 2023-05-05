# Copyright (c) 2019-2022 Ste||ar Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

macro(hpx_setup_gasnet)

include(HPX_Message)
include(FindGasnet)

find_gasnet()

if(NOT GASNET_FOUND)
  hpx_error("GASNET could not be found, please specify \
  GASNET_ROOT to point to the root of your GASNET installation"
)
endif()

endmacro()
