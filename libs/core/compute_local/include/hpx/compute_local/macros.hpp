//  Copyright (c) 2017-2019 John Biddiscombe
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>

// From file hpx/compute_local/numa_binding_allocator.hpp
#if defined(__linux) || defined(linux) || defined(__linux__)
#define NUMA_ALLOCATOR_LINUX
#endif

#if !defined(NUMA_BINDING_ALLOCATOR_DEBUG)
#define NUMA_BINDING_ALLOCATOR_DEBUG false
#endif
