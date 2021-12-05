//  Copyright (c) 2014-2015 Thomas Heller
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>

#if defined(HPX_HAVE_NETWORKING) && defined(HPX_HAVE_PARCELPORT_LCI)
#include <hpx/assert.hpp>
#include <hpx/modules/synchronization.hpp>

#include <deque>
#include <mutex>

namespace hpx::parcelset::policies::lci {
    struct tag_provider
    {
        using mutex_type = lcos::local::spinlock;

        tag_provider() noexcept
          : next_tag_(2)
        {
        }

        int acquire() noexcept
        {
            int tag = -1;
            std::lock_guard l(mtx_);
            if (free_tags_.empty())
            {
                HPX_ASSERT(next_tag_ < (std::numeric_limits<int>::max)());
                tag = next_tag_++;
            }
            else
            {
                tag = free_tags_.front();
                free_tags_.pop_front();
            }
            HPX_ASSERT(tag > 1);
            return tag;
        }

        void release(int tag)
        {
            HPX_ASSERT(tag > 1);
            std::lock_guard l(mtx_);
            HPX_ASSERT(tag < next_tag_);

            free_tags_.push_back(tag);
        }

        mutex_type mtx_;
        int next_tag_;
        std::deque<int> free_tags_;
    };
}    // namespace hpx::parcelset::policies::lci

#endif
