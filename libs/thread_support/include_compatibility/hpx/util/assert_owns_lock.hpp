//  Copyright (c) 2019 Auriane Reverdell
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#include <hpx/thread_support/config/defines.hpp>
#include <hpx/thread_support/assert_owns_lock.hpp>

#if defined(HPX_THREAD_SUPPORT_HAVE_DEPRECATION_WARNINGS)
#if defined(HPX_MSVC)
#pragma message("The header hpx/util/assert_owns_lock.hpp is deprecated, \
    please include hpx/thread_support/assert_owns_lock.hpp instead")
#else
#warning "The header hpx/util/assert_owns_lock.hpp is deprecated, \
    please include hpx/thread_support/assert_owns_lock.hpp instead"
#endif
#endif
