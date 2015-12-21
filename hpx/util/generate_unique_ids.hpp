//  Copyright (c) 2007-2012 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_UTIL_GENERATE_UNIQUE_IDS_MAR_24_2008_1014AM)
#define HPX_UTIL_GENERATE_UNIQUE_IDS_MAR_24_2008_1014AM

#include <hpx/config.hpp>
#include <hpx/runtime/naming/name.hpp>
#include <hpx/util/spinlock.hpp>

#include <boost/cstdint.hpp>
#include <boost/thread/locks.hpp>

#if defined(HPX_MSVC)
#pragma warning(push)
#pragma warning(disable: 4251)
#endif

namespace hpx { namespace util
{
    /// The unique_id_ranges class is a type responsible for generating
    /// unique ids for components, parcels, threads etc.
    class HPX_EXPORT unique_id_ranges
    {
        typedef hpx::util::spinlock mutex_type;

        mutex_type mtx_;

        /// size of the id range returned by command_getidrange
        /// FIXME: is this a policy?
        enum { range_delta = 0x100000 };

    public:
        unique_id_ranges()
          : lower_(0), upper_(0)
        {}

        /// Generate next unique component id
        naming::gid_type get_id(std::size_t count = 1);

        void set_range(
            naming::gid_type const& lower
          , naming::gid_type const& upper)
        {
            boost::lock_guard<mutex_type> l(mtx_);
            lower_ = lower;
            upper_ = upper;
        }

    private:
        /// The range of available ids for components
        naming::gid_type lower_;
        naming::gid_type upper_;
    };
}}

#if defined(HPX_MSVC)
#pragma warning(pop)
#endif

#endif


