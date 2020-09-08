//  Copyright (c) 2020 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>

#include <cstddef>

#ifdef HPX_HAVE_SPINLOCK_DEADLOCK_DETECTION
namespace hpx { namespace util { namespace detail {
    HPX_CORE_EXPORT void set_spinlock_break_on_deadlock_enabled(bool enabled);
    HPX_CORE_EXPORT bool get_spinlock_break_on_deadlock_enabled();
    HPX_CORE_EXPORT void set_spinlock_deadlock_detection_limit(
        std::size_t limit);
    HPX_CORE_EXPORT std::size_t get_spinlock_deadlock_detection_limit();
}}}    // namespace hpx::util::detail
#endif
