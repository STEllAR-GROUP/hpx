////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2013 Thomas Heller
//  Copyright (c) 2008 Peter Dimov
//  Copyright (c) 2018 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#include <hpx/config.hpp>

#include <cstddef>

#ifdef HPX_HAVE_SPINLOCK_DEADLOCK_DETECTION
namespace hpx { namespace util { namespace detail
{
    ///////////////////////////////////////////////////////////////////////////
    // We globally control whether to do spinlock deadlock detection in
    // spin-locks using this global bool variable. It will be set once by the
    // runtime configuration startup code
    HPX_EXPORT bool spinlock_break_on_deadlock = false;
    HPX_EXPORT std::size_t spinlock_deadlock_detection_limit =
        HPX_SPINLOCK_DEADLOCK_DETECTION_LIMIT;
}}}
#endif
