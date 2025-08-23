//  Copyright (c) 2016-2022 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#include <hpx/compute_local/host/target.hpp>
#include <hpx/resource_partitioner/detail/partitioner.hpp>
#include <hpx/runtime_local/get_os_thread_count.hpp>
#include <hpx/runtime_local/runtime_local.hpp>

#include <cstddef>
#include <vector>

namespace hpx::compute::host {

    std::vector<target> get_local_targets()
    {
        std::size_t const num_os_threads = hpx::get_os_thread_count();

        std::vector<target> targets;
        targets.reserve(num_os_threads);

        auto const& rp = hpx::resource::get_partitioner();
        for (std::size_t num_thread = 0; num_thread != num_os_threads;
            ++num_thread)
        {
            targets.emplace_back(rp.get_pu_mask(num_thread));
        }

        return targets;
    }
}    // namespace hpx::compute::host
