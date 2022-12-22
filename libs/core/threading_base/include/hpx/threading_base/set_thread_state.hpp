//  Copyright (c) 2007-2022 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/modules/errors.hpp>
#include <hpx/threading_base/threading_base_fwd.hpp>

namespace hpx::threads::detail {

    HPX_CORE_EXPORT thread_state set_thread_state(thread_id_type const& id,
        thread_schedule_state new_state, thread_restart_state new_state_ex,
        thread_priority priority,
        thread_schedule_hint schedulehint = thread_schedule_hint(),
        bool retry_on_active = true, error_code& ec = throws);
}    // namespace hpx::threads::detail
