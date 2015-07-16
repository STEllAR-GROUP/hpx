//  Copyright (c) 2007-2012 Hartmut Kaiser
//  Copyright (c) 2011      Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_fwd.hpp>
#include <hpx/exception.hpp>
#include <hpx/runtime/naming/resolver_client.hpp>
#include <hpx/runtime/agas/interface.hpp>
#include <hpx/util/assert.hpp>
#include <hpx/util/generate_unique_ids.hpp>
#include <hpx/util/spinlock_pool.hpp>
#include <hpx/util/unlock_guard.hpp>
#include <hpx/util/logging.hpp>

#include <boost/swap.hpp>
#include <boost/format.hpp>

namespace hpx { namespace util
{
    naming::gid_type unique_id_ranges::get_id(std::size_t count)
    {
        // create a new id
        boost::unique_lock<mutex_type> l(mtx_);

        // ensure next_id doesn't overflow
        while (!lower_ || (lower_ + count) > upper_)
        {
            lower_ = naming::invalid_gid;

            naming::gid_type lower;
            std::size_t count_ = (std::max)(std::size_t(range_delta), count);

            {
                unlock_guard<boost::unique_lock<mutex_type> > ul(l);
                lower = hpx::agas::get_next_id(count_);
            }

            // we ignore the result if some other thread has already set the
            // new lower range
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
}}

