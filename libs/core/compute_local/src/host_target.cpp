///////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2016 Thomas Heller
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
///////////////////////////////////////////////////////////////////////////////

#include <hpx/compute_local/host/target.hpp>
#include <hpx/datastructures/serialization/dynamic_bitset.hpp>
#include <hpx/resource_partitioner/detail/partitioner.hpp>
#include <hpx/runtime_local/get_os_thread_count.hpp>
#include <hpx/runtime_local/runtime_local.hpp>
#include <hpx/serialization/serialize.hpp>
#include <hpx/topology/topology.hpp>

#include <cstddef>
#include <utility>

namespace hpx::compute::host {

    std::pair<std::size_t, std::size_t> target::num_pus() const
    {
        auto const& rp = hpx::resource::get_partitioner();
        std::size_t const num_os_threads = hpx::get_os_thread_count();

        hpx::threads::mask_type const mask = native_handle().get_device();
        std::size_t const mask_size = hpx::threads::mask_size(mask);

        bool found_one = false;

        std::size_t num_thread = 0;
        for (/**/; num_thread != num_os_threads; ++num_thread)
        {
            if (hpx::threads::bit_and(
                    mask, rp.get_pu_mask(num_thread), mask_size))
            {
                found_one = true;
                break;
            }
        }

        if (!found_one)
        {
            return std::make_pair(static_cast<std::size_t>(-1), 0);
        }

        return std::make_pair(
            num_thread, (std::min)(num_os_threads, hpx::threads::count(mask)));
    }

    void target::serialize(serialization::input_archive& ar, unsigned int)
    {
        ar >> handle_.mask_;
    }

    void target::serialize(
        serialization::output_archive& ar, unsigned int) const
    {
        ar << handle_.mask_;
    }
}    // namespace hpx::compute::host
