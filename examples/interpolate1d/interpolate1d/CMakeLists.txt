# Copyright (c) 2007-2017 Hartmut Kaiser
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

add_hpx_component(
  interpolate1d INTERNAL_FLAGS
  FOLDER "Examples/Interpolate1D"
  AUTOGLOB
)

target_include_directories(interpolate1d_component PRIVATE ${HDF5_INCLUDE_DIRS})
target_link_libraries(interpolate1d_component PRIVATE ${HDF5_CXX_LIBRARIES})
