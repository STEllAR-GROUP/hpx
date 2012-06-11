//  Copyright (c) 2007-2012 Hartmut Kaiser
//  Copyright (c) 2011      Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_fwd.hpp>
#include <hpx/exception.hpp>
#include <hpx/runtime/naming/resolver_client.hpp>
#include <hpx/util/generate_unique_ids.hpp>
#include <hpx/util/spinlock_pool.hpp>
#include <hpx/util/unlock_lock.hpp>
#include <hpx/util/logging.hpp>

#include <boost/assert.hpp>
#include <boost/swap.hpp>
#include <boost/format.hpp>

namespace hpx { namespace util
{
    naming::gid_type unique_id_ranges::get_id(naming::locality const& here,
        naming::resolver_client& resolver, std::size_t count)
    {
        // create a new id
        mutex_type::scoped_lock l(mtx_);

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
      , naming::resolver_client& resolver)
    {
        mutex_type::scoped_lock al(allocation_mtx);

        const std::size_t leap_at = step / leapfrog;

        BOOST_ASSERT(leap_at != 0);

        // Get the next range of ids at the "leapfrog" point. TODO: This should
        // probably be scheduled in a new thread so that we can return to the
        // caller immediately.
        const naming::gid_type saved_i = current_i;
        if (HPX_UNLIKELY((current_lower + leap_at) == current_i))
        {
            mutex_type::scoped_lock ll(leapfrog_mtx);

            // Make sure someone hasn't already gotten a new range.
            if (((current_lower + leap_at) == saved_i) && !requested_range)
            {
                requested_range = true;

                naming::gid_type lower, upper;

                next_lower = naming::invalid_gid;
                next_upper = naming::invalid_gid;

                {
                    unlock_the_lock<mutex_type::scoped_lock> ul0(al);
                    unlock_the_lock<mutex_type::scoped_lock> ul1(ll);
                    resolver.get_id_range(here, step, lower, upper);
                }

                next_lower = lower;
                next_upper = upper;

                requested_range = false;
            }

            naming::gid_type result = current_i;
            ++current_i;
            return result;
        }

        // Check for range exhaustion.
        if (HPX_UNLIKELY((current_i + 1) > current_upper))
        {
            mutex_type::scoped_lock ll(leapfrog_mtx);

            // Sanity checks.
            if (threads::get_self_ptr())
            {
                while (HPX_UNLIKELY(!next_lower && !next_upper))
                {
                    LRT_(info) << "unique_ids::get_id: ran out of GIDs too "
                                  "quickly, possibly livelocked";

                    // Give the TM time to process the incoming response from
                    // AGAS.
                    {
                        unlock_the_lock<mutex_type::scoped_lock> ul0(al);
                        unlock_the_lock<mutex_type::scoped_lock> ul1(ll);
                        threads::get_self_ptr()->yield(threads::pending);
                    }
                }
            }

            else if (HPX_UNLIKELY(!next_lower && !next_upper))
            {
                HPX_THROW_EXCEPTION(out_of_memory,
                    "unique_ids::get_id",
                    "ran out of GIDs too quickly, definitely livelocked");
            }

            LRT_(info) << (boost::format(
                          "unique_ids::get_id: exhausted range(%1%, %2%), "
                          "switching to new range(%3%, %4%)")
                          % current_lower
                          % (current_upper - current_lower).get_lsb()
                          % next_lower
                          % (next_upper - next_lower).get_lsb());

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

