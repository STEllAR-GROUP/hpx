//  Copyright (c) 2012 Maciej Brodowicz
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/util/thread_mapper.hpp>
#include <hpx/exception.hpp>

#include <boost/format.hpp>

#include <cstring>
#include <mutex>
#include <string>

#if defined(__linux__)
#include <sys/syscall.h>
#endif

namespace hpx { namespace util
{
    ///////////////////////////////////////////////////////////////////////////
    // static members
    boost::uint32_t thread_mapper::invalid_index = static_cast<boost::uint32_t>(-1);
    long int thread_mapper::invalid_tid = -1;
    std::string thread_mapper::invalid_label;

    ///////////////////////////////////////////////////////////////////////////
    // methods
    bool thread_mapper::null_cb(boost::uint32_t) {return true;}

    long int thread_mapper::get_system_thread_id()
    {
#if defined(__linux__) && !defined(__ANDROID__) && !defined(ANDROID)
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
        if (tix >= thread_info_.size()) return false; //-V104
        thread_info_[tix].cleanup_ = boost::ref(null_cb); //-V108
        return thread_info_[tix].cleanup_(tix); //-V108
    }

    thread_mapper::~thread_mapper()
    {
        std::lock_guard<mutex_type> m(mtx_);

        for (boost::uint32_t i = 0; i < thread_info_.size(); i++) //-V104
            thread_info_[i].cleanup_(i); //-V108
    }

    boost::uint32_t thread_mapper::register_thread(char const *l, error_code& ec)
    {
        std::lock_guard<mutex_type> m(mtx_);

        boost::thread::id id = boost::this_thread::get_id();
        thread_map_type::iterator it = thread_map_.find(id);
        if (it != thread_map_.end())
        {
            // collision on boost thread ID (perhaps previous thread wasn't
            // unregistered correctly)
            unmap_thread(it);
        }

        // create mappings
        boost::uint32_t tix = thread_map_[id] =
            static_cast<boost::uint32_t>(thread_info_.size());
        thread_info_.push_back(thread_data(get_system_thread_id()));

        if (label_map_.left.find(l) != label_map_.left.end())
        {
            HPX_THROWS_IF(ec, hpx::bad_parameter,
                "hpx::thread_mapper::register_thread",
                "attempted to register thread with a duplicate label");
            return boost::uint32_t(-1);
        }

        label_map_.left.insert(label_map_type::left_value_type(l, tix));

        if (&ec != &throws)
            ec = make_success_code();

        return tix;
    }

    bool thread_mapper::unregister_thread()
    {
        std::lock_guard<mutex_type> m(mtx_);

        boost::thread::id id = boost::this_thread::get_id();
        thread_map_type::iterator it = thread_map_.find(id);
        return (it == thread_map_.end()) ? false : unmap_thread(it);
    }

    bool thread_mapper::register_callback(boost::uint32_t tix,
        callback_type const& cb)
    {
        std::lock_guard<mutex_type> m(mtx_);

        if (tix >= thread_info_.size()) return false; //-V104
        thread_info_[tix].cleanup_ = cb; //-V108
        return true;
    }

    bool thread_mapper::revoke_callback(boost::uint32_t tix)
    {
        std::lock_guard<mutex_type> m(mtx_);

        if (tix >= thread_info_.size()) return false; //-V104
        thread_info_[tix].cleanup_ = boost::ref(null_cb); //-V108
        return true;
    }

    long int thread_mapper::get_thread_id(boost::uint32_t tix) const
    {
        std::lock_guard<mutex_type> m(mtx_);

        return (tix < thread_info_.size()) ? //-V104
            thread_info_[tix].tid_: invalid_tid; //-V108
    }

    std::string const& thread_mapper::get_thread_label(boost::uint32_t tix) const
    {
        std::lock_guard<mutex_type> m(mtx_);

        label_map_type::right_map::const_iterator it = label_map_.right.find(tix);
        return (it == label_map_.right.end())? invalid_label: it->second;
    }

    boost::uint32_t thread_mapper::get_thread_index(std::string const& label) const
    {
        std::lock_guard<mutex_type> m(mtx_);

        label_map_type::left_map::const_iterator it = label_map_.left.find(label);
        return (it == label_map_.left.end())? invalid_index: it->second;
    }

    boost::uint32_t thread_mapper::get_thread_count() const
    {
        std::lock_guard<mutex_type> m(mtx_);

        return static_cast<boost::uint32_t>(thread_info_.size());
    }
}}
