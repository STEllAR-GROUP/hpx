//  Copyright (c) 2012 Maciej Brodowicz
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_UTIL_PAPI_THREADS_FEB_17_2012_0109PM)
#define HPX_UTIL_PAPI_THREADS_FEB_17_2012_0109PM

#include <hpx/hpx_fwd.hpp>

#if defined(HPX_HAVE_PAPI)

#include <hpx/exception.hpp>
#include <hpx/lcos/local_spinlock.hpp>

#include <map>
#include <vector>

#include <boost/thread.hpp>
#include <boost/noncopyable.hpp>
#include <boost/cstdint.hpp>

#include <papi.h>

namespace hpx { namespace util
{
    ///////////////////////////////////////////////////////////////////////////
    // This is the real version pluggin into PAPI
    class HPX_EXPORT papi_thread_mapper : boost::noncopyable
    {
        typedef hpx::lcos::local_spinlock mutex_type;
        typedef std::map<boost::thread::id, std::size_t> thread_map_type;

        // main lock
        mutex_type mtx_;
        // mapping from thread IDs to small integer indices
        thread_map_type thread_map_;
        // table of currently used event sets
        std::vector<std::size_t> event_sets_;

    public:
        papi_thread_mapper();
        ~papi_thread_mapper();

        // registers invoking OS thread with PAPI and assigns it a unique index
        std::size_t register_thread();

        // unregisters OS thread of index tix, stops related counters and
        // releases PAPI event set
        bool unregister_thread();

        // returns event set assigned to thread of given index, or PAPI_NULL
        // if not found
        std::size_t get_event_set(std::size_t tix);

        // returns PAPI major and minor version, sufficient to determine linking
        // compatibility of other components
        boost::uint32_t get_papi_version() const;
    };
}}

#else // HPX_HAVE_PAPI

namespace hpx { namespace util
{
    ///////////////////////////////////////////////////////////////////////////
    // This version will be used if PAPI is not available
    class HPX_EXPORT papi_thread_mapper
    {
    public:
        papi_thread_mapper() {}

        std::size_t register_thread() { return 0; }
        bool unregister_thread() { return true; }
    };
}}

#endif

#endif
