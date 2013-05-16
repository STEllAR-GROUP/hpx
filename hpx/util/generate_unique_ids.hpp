//  Copyright (c) 2007-2012 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_UTIL_GENERATE_UNIQUE_IDS_MAR_24_2008_1014AM)
#define HPX_UTIL_GENERATE_UNIQUE_IDS_MAR_24_2008_1014AM

#include <boost/thread.hpp>
#include <boost/assert.hpp>
#include <boost/cstdint.hpp>

#include <hpx/hpx_fwd.hpp>
#include <hpx/config.hpp>
#include <hpx/util/spinlock.hpp>
#include <hpx/exception.hpp>

#if defined(BOOST_MSVC)
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
        enum { range_delta = 16384 };

    public:
        unique_id_ranges()
          : lower_(0), upper_(0)
        {}

        /// Generate next unique component id
        naming::gid_type get_id(naming::locality const& here,
            naming::resolver_client& resolver, std::size_t count = 1);

        /// Not thread-safe
        void set_range(
            naming::gid_type const& lower
          , naming::gid_type const& upper)
        {
            lower_ = lower;
            upper_ = upper;
        }

    private:
        /// The range of available ids for components
        naming::gid_type lower_;
        naming::gid_type upper_;
    };

    /// The unique_ids class is a restricted form of unique_id_ranges, which
    /// only allocates one gid at a time.
    struct HPX_EXPORT unique_ids
    {
        typedef hpx::util::spinlock mutex_type;

      private:
        mutex_type leapfrog_mtx;
        mutex_type allocation_mtx;

        naming::gid_type current_lower;
        naming::gid_type current_i;
        naming::gid_type current_upper;
        naming::gid_type next_lower;
        naming::gid_type next_upper;
        bool requested_range;
        const std::size_t step;
        const std::size_t leapfrog;

      public:
        unique_ids(
            std::size_t step_ = HPX_INITIAL_GID_RANGE
          , std::size_t leapfrog_ = 4
            )
          : current_lower(0)
          , current_i(0)
          , current_upper(0)
          , next_lower(0)
          , next_upper(0)
          , requested_range(false)
          , step(step_)
          , leapfrog(leapfrog_)
        {
            BOOST_ASSERT(leapfrog_ != 0);
        }

        /// Generate next unique id
        naming::gid_type get_id(
            naming::locality const& here,
            naming::resolver_client& resolver
            );

        /// Not thread-safe
        void set_range(
            naming::gid_type const& lower
          , naming::gid_type const& upper
            )
        {
            mutex_type::scoped_lock al(allocation_mtx);
            current_lower = lower;
            current_i = lower;
            current_upper = upper;
        }
    };
}}

#if defined(BOOST_MSVC)
#pragma warning(pop)
#endif

#endif


