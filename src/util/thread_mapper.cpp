//  Copyright (c) 2012 Maciej Brodowicz
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_fwd.hpp>
#include <hpx/util/thread_mapper.hpp>

#include <boost/format.hpp>

#if defined(__linux__)
#include <sys/syscall.h>
#endif

namespace hpx { namespace util
{
    ///////////////////////////////////////////////////////////////////////////
    // methods
    bool thread_mapper::null_cb(boost::uint32_t) {return true;}

    unsigned long thread_mapper::get_papi_thread_id()
    {
#if defined(__linux__)
        // this has been tested only on x86_*
        return syscall(SYS_gettid);
#else
        return invalid_tid;
#endif
    }

    bool thread_mapper::unmap_thread(thread_map_type::iterator& it)
    {
        boost::uint32_t tix = it->second;
        thread_map_.erase(it);
        if (tix >= thread_info_.size()) return false;
        thread_info_[tix].cleanup_ = boost::ref(null_cb);
        return thread_info_[tix].cleanup_(tix);
    }

    thread_mapper::~thread_mapper()
    {
        mutex_type::scoped_lock m(mtx_);

        for (boost::uint32_t i = 0; i < thread_info_.size(); i++)
            thread_info_[i].cleanup_(i);
    }

    boost::uint32_t thread_mapper::register_thread(char const *l)
    {
        mutex_type::scoped_lock m(mtx_);

        boost::thread::id id = boost::this_thread::get_id();
        thread_map_type::iterator it = thread_map_.find(id);
        if (it != thread_map_.end())
        {   // collision on boost thread ID (perhaps previous thread wasn't
            // unregistered correctly)
            unmap_thread(it);
        }
        // increment category count and create mappings
        boost::uint32_t tix = thread_map_[id] =
            static_cast<boost::uint32_t>(thread_info_.size());
        thread_info_.push_back(
            thread_data(l, label_count_[l], get_papi_thread_id()));
        label_map_[thread_id(l, label_count_[l]++)] = tix;
        return tix;
    }

    bool thread_mapper::unregister_thread()
    {
        mutex_type::scoped_lock m(mtx_);

        boost::thread::id id = boost::this_thread::get_id();
        thread_map_type::iterator it = thread_map_.find(id);
        return (it == thread_map_.end())? false: unmap_thread(it);
    }

    bool thread_mapper::register_callback(boost::uint32_t tix,
                                               callback_type cb)
    {
        mutex_type::scoped_lock m(mtx_);

        if (tix >= thread_info_.size()) return false;
        thread_info_[tix].cleanup_ = cb;
        return true;
    }

    bool thread_mapper::revoke_callback(boost::uint32_t tix)
    {
        mutex_type::scoped_lock m(mtx_);

        if (tix >= thread_info_.size()) return false;
        thread_info_[tix].cleanup_ = boost::ref(null_cb);
        return true;
    }

    unsigned long thread_mapper::get_thread_id(boost::uint32_t tix) const
    {
        mutex_type::scoped_lock m(mtx_);

        return (tix < thread_info_.size())? thread_info_[tix].tid_: invalid_tid;
    }

    char const *thread_mapper::get_thread_label(boost::uint32_t tix) const
    {
        mutex_type::scoped_lock m(mtx_);

        return (tix < thread_info_.size())? thread_info_[tix].label_: 0;
    }

    boost::uint32_t thread_mapper::get_thread_instance(boost::uint32_t tix) const
    {
        mutex_type::scoped_lock m(mtx_);

        return (tix < thread_info_.size())? thread_info_[tix].instance_: invalid_index;
    }

    boost::uint32_t thread_mapper::get_thread_index(char const *label,
                                                       boost::uint32_t inst) const
    {
        mutex_type::scoped_lock m(mtx_);

        thread_id tid(label, inst);
        label_map_type::const_iterator it = label_map_.find(tid);
        return (it == label_map_.end())? invalid_index: it->second;
    }

    boost::uint32_t thread_mapper::get_thread_count() const
    {
        mutex_type::scoped_lock m(mtx_);

        return static_cast<boost::uint32_t>(thread_info_.size());
    }

    void thread_mapper::get_registered_labels(std::vector<char const *>& labels) const
    {
        mutex_type::scoped_lock m(mtx_);

        label_count_type::const_iterator it;
        for (it = label_count_.begin(); it != label_count_.end(); ++it)
            labels.push_back(it->first);
    }
}}
