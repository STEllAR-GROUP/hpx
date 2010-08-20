//  Copyright (c) 2007-2010 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_UTIL_GENERATE_UNIQUE_IDS_MAR_24_2008_1014AM)
#define HPX_UTIL_GENERATE_UNIQUE_IDS_MAR_24_2008_1014AM

#include <boost/thread.hpp>
#include <boost/assert.hpp>
#include <boost/cstdint.hpp>

#include <hpx/config.hpp>
#include <hpx/runtime/naming/resolver_client.hpp>
#include <hpx/util/spinlock_pool.hpp>
#include <hpx/util/unlock_lock.hpp>

namespace hpx { namespace util
{
    /// The unique_ids class is a type responsible for generating 
    /// unique ids for components, parcels, threads etc.
    class unique_ids 
    {
        struct tag {};
        typedef hpx::util::spinlock_pool<tag> mutex_type;

        /// size of the id range returned by command_getidrange
        /// FIXME: is this a policy?
        enum { range_delta = 16384 };

    public:
        unique_ids()
          : lower_(0), upper_(0)
        {}

        /// Generate next unique component id
        naming::gid_type get_id(naming::locality const& here,
            naming::resolver_client const& resolver, std::size_t count = 1)
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

    private:
        /// The range of available ids for components
        naming::gid_type lower_;
        naming::gid_type upper_;
    };
    
}}

#endif


