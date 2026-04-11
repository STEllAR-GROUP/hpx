//  Copyright (c) 2024-2026 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/modules/errors.hpp>
#include <hpx/threading_base/threading_base_fwd.hpp>

#include <cstdint>

#include <hpx/config/warnings_prefix.hpp>

namespace hpx::threads {

    HPX_CXX_EXPORT HPX_CORE_EXPORT thread_state set_thread_affinity(
        thread_id_type const& id, std::int16_t target_pu,
        thread_priority priority = thread_priority::bound,
        error_code& ec = throws);

}    // namespace hpx::threads

namespace hpx::this_thread {

    HPX_CXX_EXPORT HPX_CORE_EXPORT void set_affinity(std::int16_t target_pu,
        threads::thread_priority priority = threads::thread_priority::bound,
        error_code& ec = throws);

}    // namespace hpx::this_thread

#include <hpx/config/warnings_suffix.hpp>
