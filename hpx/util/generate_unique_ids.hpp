//  Copyright (c) 2007-2008 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_UTIL_GENERATE_UNIQUE_IDS_MAR_24_2008_1014AM)
#define HPX_UTIL_GENERATE_UNIQUE_IDS_MAR_24_2008_1014AM

#include <boost/thread.hpp>
#include <boost/assert.hpp>
#include <boost/cstdint.hpp>
#include <boost/utility/singleton.hpp>

#include <hpx/config.hpp>
#include <hpx/components/component_type.hpp>

#if HPX_USE_TBB != 0
#include <tbb/atomic.h>
#endif

namespace hpx { namespace util
{
    /// The unique_ids class is a singleton type responsible for generating 
    /// unique ids for components, parcels, px_threads etc.
    class unique_ids : public boost::singleton<unique_ids>
    {
#if HPX_USE_TBB == 0
        typedef boost::mutex mutex_type;
#endif

    public:
        unique_ids(boost::restricted)
        {
            next_id_ = 0;
            next_thread_id_ = 0;
        }

        /// Generate next unique component id
        naming::id_type get_id()
        {
            // create a new id
#if HPX_USE_TBB == 0
            mutex_type::scoped_lock l(mtx_);
#endif            
            // ensure next_id doesn't overflow
            BOOST_ASSERT(0 == (~0xFFFFFFFFFFFFLL & next_id_));
            return ++next_id_;
        }
        
        /// Generate unique thread id
        naming::id_type get_thread_id()
        {
            // create a new id
#if HPX_USE_TBB == 0
            mutex_type::scoped_lock l(mtx_);
#endif            
            // ensure next_id doesn't overflow
            BOOST_ASSERT(0 == (~0xFFFFFFFFFFLL & next_thread_id_));
            
            static boost::int64_t const px_thread_id_1 = 
                components::component_px_thread << 24;
            static boost::int64_t const px_thread_id = px_thread_id_1 << 16;

            return px_thread_id | ++next_thread_id_;
        }

    private:
        /// The next available id for components
#if HPX_USE_TBB == 0
        mutex_type mtx_;
        boost::uint64_t next_id_;
        boost::uint64_t next_thread_id_;
#else
        tbb::atomic<boost::uint64_t> next_id_;
        tbb::atomic<boost::uint64_t> next_thread_id_;
#endif
    };
    
}}

#endif


