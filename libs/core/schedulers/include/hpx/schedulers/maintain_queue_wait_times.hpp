//  Copyright (c) 2005-2022 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Adelstein-Lelbach
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>

#ifdef HPX_HAVE_THREAD_QUEUE_WAITTIME
namespace hpx::threads::policies {

    HPX_CXX_CORE_EXPORT HPX_CORE_EXPORT void
    set_maintain_queue_wait_times_enabled(bool enabled) noexcept;
    HPX_CXX_CORE_EXPORT HPX_CORE_EXPORT bool
    get_maintain_queue_wait_times_enabled() noexcept;
}    // namespace hpx::threads::policies
#endif
