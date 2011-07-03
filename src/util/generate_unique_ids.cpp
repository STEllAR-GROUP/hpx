//  Copyright (c) 2007-2011 Hartmut Kaiser
//  Copyright (c) 2011      Bryce Lelbach 
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <boost/assert.hpp>
#include <boost/swap.hpp>

#include <hpx/runtime/naming/resolver_client.hpp>
#include <hpx/util/generate_unique_ids.hpp>
#include <hpx/util/spinlock_pool.hpp>
#include <hpx/util/unlock_lock.hpp>

namespace hpx { namespace util
{
    naming::gid_type unique_id_ranges::get_id(naming::locality const& here,
        naming::resolver_client& resolver, std::size_t count)
    {
        // create a new id
        mutex_type::scoped_lock l(this);

        // ensure next_id doesn't overflow
        if ((lower_ + count) > upper_) 
        {
            naming::gid_type lower;
            naming::gid_type upper;

            {
                unlock_the_lock<mutex_type::scoped_lock> ul(l);
                resolver.get_id_range(here, 
                    (std::max)(std::size_t(range_delta), count), 
                    lower, upper);
            }

            lower_ = lower;
            upper_ = upper;
        }

        naming::gid_type result = lower_;
        lower_ += count;
        return result;
    }

    naming::gid_type unique_ids::get_id(
        naming::locality const& here
      , naming::resolver_client& resolver
    ) {
        mutex_type::scoped_lock l(this);
          
        const std::size_t leap_at = step / leapfrog;

        BOOST_ASSERT(leap_at != 0);

        // Get the next range of ids at the "leapfrog" point. TODO: This should
        // probably be scheduled in a new thread so that we can return to the
        // caller immediately.
        if ((current_lower + leap_at) == current_i)
        {
            naming::gid_type lower, upper;

            {
                unlock_the_lock<mutex_type::scoped_lock> ul(l);
                resolver.get_id_range(here, step, lower, upper);
            }

            next_lower = lower;
            next_upper = upper;
        }

        // Check for range exhaustion.
        if ((current_i + 1) > current_upper) 
        {
            // Sanity checks.
            BOOST_ASSERT(next_lower);
            BOOST_ASSERT(next_upper);
            // FIXME: Doesn't work as GID subtraction isn't implemented.
            //BOOST_ASSERT((next_upper - next_lower) == step);

            // Switch to the next range.
            boost::swap(current_lower, next_lower);
            current_i = current_lower;
            boost::swap(current_upper, next_upper);
        }

        naming::gid_type result = current_i;
        ++current_i;
        return result;
    }
}}



