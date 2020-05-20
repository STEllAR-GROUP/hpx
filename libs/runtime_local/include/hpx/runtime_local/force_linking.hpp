//  Copyright (c) 2020 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>

#include <cstdint>

namespace hpx { namespace runtime_local {

    struct force_linking_helper
    {
#if defined(HPX_HAVE_THREAD_AWARE_TIMER_COMPATIBILITY)
        std::uint64_t (*take_time_stamp)();
#endif
    };

    force_linking_helper& force_linking();
}}    // namespace hpx::runtime_local
