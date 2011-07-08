//  Copyright (c) 2007-2011 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Lelbach
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_UTIL_GENERATE_UNIQUE_IDS_MAR_24_2008_1014AM)
#define HPX_UTIL_GENERATE_UNIQUE_IDS_MAR_24_2008_1014AM

#include <hpx/hpx_fwd.hpp>

#include <boost/thread.hpp>
#include <boost/assert.hpp>
#include <boost/cstdint.hpp>

namespace hpx { namespace util
{
    /// The unique_ids class is a type responsible for generating 
    /// unique ids for components, parcels, threads etc.
    class HPX_EXPORT unique_ids 
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
            naming::resolver_client& resolver, std::size_t count = 1);

        void set_range(naming::gid_type const& lower, 
            naming::gid_type const& upper) 
        {
            lower_ = lower;
            upper_ = upper;
        }

    private:
        /// The range of available ids for components
        naming::gid_type lower_;
        naming::gid_type upper_;
    };

    typedef unique_ids unique_id_ranges;
}}

#endif


