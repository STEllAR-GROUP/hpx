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
#include <hpx/execution_base/detail/spinlock_deadlock_detection.hpp>

#include <cstddef>

#ifdef HPX_HAVE_SPINLOCK_DEADLOCK_DETECTION

namespace hpx::util::detail {

    static bool spinlock_break_on_deadlock_enabled = false;
    static std::size_t spinlock_deadlock_detection_limit =
        HPX_SPINLOCK_DEADLOCK_DETECTION_LIMIT;

    void set_spinlock_break_on_deadlock_enabled(bool enabled) noexcept
    {
        spinlock_break_on_deadlock_enabled = enabled;
    }

    bool get_spinlock_break_on_deadlock_enabled() noexcept
    {
        return spinlock_break_on_deadlock_enabled;
    }

    void set_spinlock_deadlock_detection_limit(std::size_t limit) noexcept
    {
        spinlock_deadlock_detection_limit = limit;
    }

    std::size_t get_spinlock_deadlock_detection_limit() noexcept
    {
        return spinlock_deadlock_detection_limit;
    }
}    // namespace hpx::util::detail
#endif
