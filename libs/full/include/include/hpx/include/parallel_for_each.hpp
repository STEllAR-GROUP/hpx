//  Copyright (c) 2007-2015 Hartmut Kaiser
//  Copyright (c) 2014 Grant Mercer
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/parallel/algorithms/for_each.hpp>
#include <hpx/parallel/container_algorithms/for_each.hpp>

#if defined(HPX_HAVE_DISTRIBUTED_RUNTIME)
#include <hpx/parallel/segmented_algorithms/for_each.hpp>
#endif
