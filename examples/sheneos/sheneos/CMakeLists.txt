# Copyright (c) 2007-2023 Hartmut Kaiser
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

add_hpx_component(
  sheneos
  INSTALL_COMPONENT examples
  DEPENDENCIES ${HDF5_LIBRARIES}
  FOLDER "Examples/ShenEOS"
  AUTOGLOB
)
