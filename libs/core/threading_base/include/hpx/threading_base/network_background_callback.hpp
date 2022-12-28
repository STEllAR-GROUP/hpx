//  Copyright (c) 2019 Mikael Simberg
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/functional/function.hpp>

#include <cstddef>
#include <cstdint>

namespace hpx::threads::detail {

#if defined(HPX_HAVE_BACKGROUND_THREAD_COUNTERS) &&                            \
    defined(HPX_HAVE_THREAD_IDLE_RATES)
    using network_background_callback_type =
        hpx::function<bool(std::size_t, std::int64_t&, std::int64_t&)>;
#else
    using network_background_callback_type = hpx::function<bool(std::size_t)>;
#endif
}    // namespace hpx::threads::detail
