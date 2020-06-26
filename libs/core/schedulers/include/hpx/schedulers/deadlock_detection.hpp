//  Copyright (c) 2005-2017 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Adelstein-Lelbach
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>

namespace hpx { namespace threads { namespace policies {
#ifdef HPX_HAVE_THREAD_MINIMAL_DEADLOCK_DETECTION
    HPX_CORE_EXPORT void set_minimal_deadlock_detection_enabled(bool enabled);
    HPX_CORE_EXPORT bool get_minimal_deadlock_detection_enabled();
#endif
}}}    // namespace hpx::threads::policies
