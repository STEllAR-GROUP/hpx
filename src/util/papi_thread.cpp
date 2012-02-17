//  Copyright (c) 2012 Maciej Brodowicz
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/util/papi_threads.hpp>

#if defined(HPX_HAVE_PAPI)

namespace hpx { namespace util
{
    papi_thread_mapper::papi_thread_mapper()
    {
        if (PAPI_is_initialized() != PAPI_NOT_INITED)
        {
            if (PAPI_library_init(PAPI_VER_CURRENT) != PAPI_VER_CURRENT)
            {
                HPX_THROW_EXCEPTION(hpx::no_success,
                    "papi_thread_support()",
                    "PAPI library initialization failed (version mismatch)");
            }
        }
    }

    papi_thread_mapper::~papi_thread_mapper()
    {
        // FIXME: free all resources allocated in constructor
    }

    std::size_t papi_thread_mapper::register_thread()
    {
        mutex_type::scoped_lock m(mtx_);

        boost::thread::id id = boost::this_thread::get_id();
        thread_map_type::iterator it = thread_map_.find(id);
        // repeated registrations are ignored
        if (it != thread_map_.end() && event_sets_[it->second] != PAPI_NULL)
            return it->second;

        int evset = PAPI_NULL;
        if (PAPI_create_eventset(&evset) != PAPI_OK)
        {
            HPX_THROW_EXCEPTION(hpx::no_success,
                "papi_thread_support::register_thread()",
                "creation of event set failed");
        }
        event_sets_.push_back(evset);
        return thread_map_[id] = event_sets_.size()-1;
    }

    bool papi_thread_mapper::unregister_thread()
    {
        mutex_type::scoped_lock m(mtx_);

        boost::thread::id id = boost::this_thread::get_id();
        thread_map_type::iterator it = thread_map_.find(id);
        if (it != thread_map_.end())
            return false;

        std::size_t tix = event_sets_[it->second];
        if (tix >= event_sets_.size()) return false;
        if (PAPI_stop(event_sets_[tix]) != PAPI_OK ||
            PAPI_cleanup_eventset(event_sets_[tix]) != PAPI_OK ||
            PAPI_destroy_eventset(&event_sets_[tix]) != PAPI_OK)
        {
            return false;
        }
        // at this point event set is set to PAPI_NULL
        return true;
    }

    std::size_t papi_thread_mapper::get_event_set(std::size_t tix)
    {
        mutex_type::scoped_lock m(mtx_);

        if (tix >= event_sets_.size()) return PAPI_NULL;
        return event_sets_[tix];
    }

    boost::uint32_t papi_thread_mapper::get_papi_version() const
    {
        return PAPI_VER_CURRENT;
    }
}}

#endif
