//  Copyright (c) 2007-2011 Hartmut Kaiser
//  Copyright (c) 2011      Bryce Lelbach 
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/version.hpp>

#if HPX_AGAS_VERSION <= 0x10

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/naming/resolver_client.hpp>
#include <hpx/util/generate_unique_ids.hpp>
#include <hpx/util/spinlock_pool.hpp>
#include <hpx/util/unlock_lock.hpp>

#include <boost/assert.hpp>

namespace hpx { namespace util
{
    naming::gid_type unique_ids::get_id(naming::locality const& here,
        naming::resolver_client& resolver, std::size_t count)
    {
        // create a new id
        mutex_type::scoped_lock l(this);

        // ensure next_id doesn't overflow
        if (lower_ + count > upper_) 
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
}}

#endif
