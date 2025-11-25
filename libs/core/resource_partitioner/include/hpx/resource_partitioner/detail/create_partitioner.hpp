//  Copyright (c)      2017 Shoshana Jakobovits
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/modules/ini.hpp>
#include <hpx/resource_partitioner/partitioner_fwd.hpp>

namespace hpx::resource::detail {

    HPX_CORE_EXPORT partitioner& create_partitioner(
        resource::partitioner_mode rpmode, hpx::util::section const& rtcfg,
        hpx::threads::policies::detail::affinity_data const& affinity_data);
}    // namespace hpx::resource::detail
