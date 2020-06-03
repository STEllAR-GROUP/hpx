//  Copyright (c) 2020 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#include <hpx/runtime_local/force_linking.hpp>
#include <hpx/runtime_local/os_thread_type.hpp>

#if defined(HPX_HAVE_THREAD_AWARE_TIMER_COMPATIBILITY)
#include <hpx/util/thread_aware_timer.hpp>
#endif

namespace hpx { namespace runtime_local {
    force_linking_helper& force_linking()
    {
        static force_linking_helper helper
        {
            &hpx::runtime_local::get_os_thread_type_name,
#if defined(HPX_HAVE_THREAD_AWARE_TIMER_COMPATIBILITY)
                &hpx::util::thread_aware_timer::take_time_stamp
#endif
        };
        return helper;
    }
}}    // namespace hpx::runtime_local
