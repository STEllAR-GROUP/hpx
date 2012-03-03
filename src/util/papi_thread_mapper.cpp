//  Copyright (c) 2012 Maciej Brodowicz
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_fwd.hpp>
#include <hpx/util/papi_thread_mapper.hpp>

#include <boost/format.hpp>

#if defined(HPX_HAVE_PAPI)

#include <papi.h>

namespace hpx { namespace util
{
    ///////////////////////////////////////////////////////////////////////////
    // methods
    bool papi_thread_mapper::destroy_event_set(int& evset)
    {
        if (evset == PAPI_NULL) return true;
        bool rc = false;
        int n = 0;
        if (PAPI_list_events(evset, 0, &n) == PAPI_OK)
        {
            std::vector<long long> tmp(n);
            rc = (PAPI_stop(evset, &tmp[0]) == PAPI_OK) &&
                 (PAPI_cleanup_eventset(evset) == PAPI_OK) &&
                 (PAPI_destroy_eventset(&evset) == PAPI_OK);
        }
        return rc;
    }

    papi_thread_mapper::thread_data::thread_data(): evset_(PAPI_NULL) { }

    papi_thread_mapper::papi_thread_mapper()
    {
        if (PAPI_is_initialized() == PAPI_NOT_INITED)
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
        mutex_type::scoped_lock m(mtx_);

        for (std::size_t i = 0; i < thread_info_.size(); i++)
            destroy_event_set(thread_info_[i].evset_);
    }

    std::size_t papi_thread_mapper::register_thread(char const *l)
    {
        mutex_type::scoped_lock m(mtx_);

        boost::thread::id id = boost::this_thread::get_id();
        thread_map_type::iterator it = thread_map_.find(id);
        // repeated registrations are ignored
        if (it != thread_map_.end() &&
            thread_info_[it->second].evset_ != PAPI_NULL)
            return it->second;

        int evset = PAPI_NULL;
        if (PAPI_create_eventset(&evset) != PAPI_OK)
        {
            HPX_THROW_EXCEPTION(hpx::no_success,
                "papi_thread_support::register_thread()",
                "creation of event set failed");
        }
        // create mappings
        std::size_t tix = thread_map_[id] = thread_info_.size();
        // generate unique label
        label_map_type::iterator li = label_map_.find(l);
        if (li == label_map_.end()) label_map_[l] = 0;
        boost::format lfmt("%s#%d");
        thread_info_.push_back(thread_data(str(lfmt % l % label_map_[l]), evset));
        label_map_[l]++;
        return tix;
    }

    bool papi_thread_mapper::unregister_thread()
    {
        mutex_type::scoped_lock m(mtx_);

        boost::thread::id id = boost::this_thread::get_id();
        thread_map_type::iterator it = thread_map_.find(id);
        if (it != thread_map_.end())
            return false;

        std::size_t tix = it->second;
        if (tix >= thread_info_.size()) return false;
        return destroy_event_set(thread_info_[tix].evset_);
    }

    int papi_thread_mapper::get_event_set(std::size_t tix) const
    {
        mutex_type::scoped_lock m(mtx_);

        return (tix < thread_info_.size())? thread_info_[tix].evset_: PAPI_NULL;
    }

    std::string const& papi_thread_mapper::get_thread_label(std::size_t tix) const
    {
        mutex_type::scoped_lock m(mtx_);

        return (tix < thread_info_.size())? thread_info_[tix].label_: empty_label_;
    }

    std::size_t papi_thread_mapper::get_thread_count() const
    {
        mutex_type::scoped_lock m(mtx_);

        return thread_info_.size();
    }

    boost::uint32_t papi_thread_mapper::get_papi_version() const
    {
        return PAPI_VER_CURRENT;
    }
}}

#endif
