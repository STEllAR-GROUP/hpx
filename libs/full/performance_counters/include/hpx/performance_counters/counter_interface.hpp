//  Copyright (c) 2021 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/futures/future.hpp>
#include <hpx/modules/errors.hpp>
#include <hpx/naming_base/id_type.hpp>
#include <hpx/performance_counters/counters.hpp>

namespace hpx { namespace performance_counters {

    ///////////////////////////////////////////////////////////////////////////
    HPX_EXPORT hpx::future<id_type> create_performance_counter_async(
        id_type target_id, counter_info const& info);

    inline id_type create_performance_counter(
        id_type target_id, counter_info const& info, error_code& ec = throws)
    {
        return create_performance_counter_async(target_id, info).get(ec);
    }
}}    // namespace hpx::performance_counters
