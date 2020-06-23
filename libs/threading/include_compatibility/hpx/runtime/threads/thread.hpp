//  Copyright (c) 2019 STE||AR Group
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/threading/config/defines.hpp>
#include <hpx/thread.hpp>

#if HPX_THREADING_HAVE_DEPRECATION_WARNINGS
#if defined(HPX_MSVC)
#pragma message("The header hpx/runtime/threads/thread.hpp is deprecated, \
    please include hpx/thread.hpp instead")
#else
#warning "The header hpx/runtime/threads/thread.hpp is deprecated, \
    please include hpx/thread.hpp instead"
#endif
#endif
