//  Copyright (c) 2012 Maciej Brodowicz
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_UTIL_PAPI_THREADS_MAPPER_FEB_17_2012_0109PM)
#define HPX_UTIL_PAPI_THREAD_MAPPER_FEB_17_2012_0109PM

#include <hpx/hpx_fwd.hpp>

#if defined(HPX_HAVE_PAPI)

#include <hpx/exception.hpp>
#include <hpx/lcos/local/spinlock.hpp>

#include <map>
#include <vector>
#include <string>

#include <boost/thread.hpp>
#include <boost/noncopyable.hpp>
#include <boost/cstdint.hpp>

namespace hpx { namespace util
{
    ///////////////////////////////////////////////////////////////////////////
    // PAPI support and early initialization
    class HPX_EXPORT papi_thread_mapper : boost::noncopyable
    {
        typedef hpx::lcos::local::spinlock mutex_type;
        typedef std::map<boost::thread::id, std::size_t> thread_map_type;
        typedef std::map<char const *, std::size_t> label_map_type;

        // thread-specific data
        struct thread_data
        {
            std::string label_;
            int evset_;

            thread_data();
            thread_data(std::string const& s, int e): label_(s), evset_(e) { }
        };

        // main lock
        mutable mutex_type mtx_;
        // mapping from boost thread IDs to small integer indices
        thread_map_type thread_map_;
        // thread counts for each thread category
        label_map_type label_map_;
        // table of thread specific data
        std::vector<thread_data> thread_info_;
        // empty label for undefined accesses
        std::string const empty_label_;


    protected:
        // release resources associated with a specific PAPI event set
        bool destroy_event_set(int& evset);
        
    public:
        papi_thread_mapper();
        ~papi_thread_mapper();

        // registers invoking OS thread with PAPI and assigns it a unique index
        std::size_t register_thread(char const *label = "unspecified-thread");

        // unregisters OS thread of index tix, stops related counters and
        // releases PAPI event set
        bool unregister_thread();

        // returns event set assigned to thread of given index, or PAPI_NULL
        // if not found
        int get_event_set(std::size_t tix) const;

        // returns the label of thread tix
        std::string const& get_thread_label(std::size_t tix) const;
        
        // returns the number of threads registered so far
        std::size_t get_thread_count() const;

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

        std::size_t register_thread(char const *) { return 0; }
        bool unregister_thread() { return true; }
    };
}}

#endif

#endif
