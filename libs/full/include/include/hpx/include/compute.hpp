//  Copyright (c) 2016 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#if defined(__CUDACC__) || defined(HPX_HAVE_MODULE_COMPUTE_CUDA)
#include <hpx/compute/cuda.hpp>
#endif
#include <hpx/compute/host.hpp>
#include <hpx/compute/serialization/vector.hpp>
#include <hpx/compute/vector.hpp>
