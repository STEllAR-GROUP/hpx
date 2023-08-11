# Copyright (c) 2021-2023 Hartmut Kaiser
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

if(HPX_WITH_DISTRIBUTED_RUNTIME)
  hpx_debug("creating static_parcelports.hpp: "
            ${HPX_STATIC_PARCELPORT_PLUGINS}
  )

  # handle parcelports module to create proper dependencies
  add_subdirectory(libs/full/parcelports)
endif()
