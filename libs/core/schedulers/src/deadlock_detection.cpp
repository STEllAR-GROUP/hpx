//  Copyright (c) 2005-2022 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Adelstein-Lelbach
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>

#ifdef HPX_HAVE_THREAD_MINIMAL_DEADLOCK_DETECTION
#include <hpx/schedulers/deadlock_detection.hpp>

namespace hpx::threads::policies {

    static bool minimal_deadlock_detection_enabled = false;

    void set_minimal_deadlock_detection_enabled(bool enabled) noexcept
    {
        minimal_deadlock_detection_enabled = enabled;
    }

    bool get_minimal_deadlock_detection_enabled() noexcept
    {
        return minimal_deadlock_detection_enabled;
    }
}    // namespace hpx::threads::policies

#endif
