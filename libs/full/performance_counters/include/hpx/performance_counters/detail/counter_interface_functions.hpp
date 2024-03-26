//  Copyright (c) 2021 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/futures/future_fwd.hpp>
#include <hpx/naming_base/id_type.hpp>
#include <hpx/performance_counters/counters.hpp>

namespace hpx { namespace performance_counters { namespace detail {

    ///////////////////////////////////////////////////////////////////////////
    extern HPX_EXPORT hpx::future<id_type> (*create_performance_counter_async)(
        id_type target_id, counter_info const& info);
}}}    // namespace hpx::performance_counters::detail
