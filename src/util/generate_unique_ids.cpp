//  Copyright (c) 2007-2012 Hartmut Kaiser
//  Copyright (c) 2011      Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_fwd.hpp>
#include <hpx/exception.hpp>
#include <hpx/runtime/naming/resolver_client.hpp>
#include <hpx/runtime/agas/interface.hpp>
#include <hpx/util/generate_unique_ids.hpp>
#include <hpx/util/spinlock_pool.hpp>
#include <hpx/util/unlock_lock.hpp>
#include <hpx/util/logging.hpp>

#include <boost/assert.hpp>
#include <boost/swap.hpp>
#include <boost/format.hpp>

namespace hpx { namespace util
{
    naming::gid_type unique_id_ranges::get_id(std::size_t count)
    {
        // create a new id
        mutex_type::scoped_lock l(mtx_);

        // ensure next_id doesn't overflow
        if ((lower_ + count) > upper_)
        {
            naming::gid_type lower;
            naming::gid_type upper;

            std::size_t count_ = (std::max)(std::size_t(range_delta), count);

            {
                unlock_the_lock<mutex_type::scoped_lock> ul(l);
                lower = hpx::agas::get_next_id(count_);
            }

            lower_ = lower;
            upper_ = lower + count_;
        }

        naming::gid_type result = lower_;
        lower_ += count;
        return result;
    }
}}

