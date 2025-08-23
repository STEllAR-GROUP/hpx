//  Copyright (c) 2007-2024 Hartmut Kaiser
//  Copyright (c) 2011      Bryce Lelbach
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#include <hpx/components_base/agas_interface.hpp>
#include <hpx/components_base/generate_unique_ids.hpp>
#include <hpx/thread_support/unlock_guard.hpp>

#include <algorithm>
#include <cstddef>
#include <mutex>

namespace hpx::util {

    naming::gid_type unique_id_ranges::get_id(std::size_t count)
    {
        // create a new id
        std::unique_lock l(mtx_);

        // ensure next_id doesn't overflow
        while (!lower_ || (lower_ + count) > upper_)
        {
            lower_ = naming::invalid_gid;

            naming::gid_type lower;
            std::size_t const count_ = (std::max) (range_delta, count);

            {
                hpx::unlock_guard ul(l);
                lower = hpx::agas::get_next_id(count_);
            }

            // we ignore the result if some other thread has already set the new
            // lower range
            if (!lower_)
            {
                lower_ = lower;
                upper_ = lower + count_;
            }
        }

        naming::gid_type result = lower_;
        lower_ += count;
        return result;
    }

    void unique_id_ranges::set_range(
        naming::gid_type const& lower, naming::gid_type const& upper)
    {
        std::lock_guard l(mtx_);
        lower_ = lower;
        upper_ = upper;
    }
}    // namespace hpx::util
