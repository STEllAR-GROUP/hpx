# Copyright (c) 2016 Hartmut Kaiser
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

# Locate the Boost.SIMD template library.
# Boost.SIMD can be found at https://github.com/NumScale/boost.simd

if(NOT BOOST_SIMD_ROOT)
  hpx_error("Using Boost.SIMD requires to set the variable BOOST_SIMD_ROOT to the Boost.SIMD installation directory.")
endif()

include_directories(SYSTEM ${BOOST_SIMD_ROOT}/include)

hpx_add_config_define(HPX_HAVE_DATAPAR)
hpx_add_config_define(HPX_HAVE_DATAPAR_BOOST_SIMD)

hpx_info("Found Boost.SIMD (vectorization): " ${BOOST_SIMD_ROOT}/include)
