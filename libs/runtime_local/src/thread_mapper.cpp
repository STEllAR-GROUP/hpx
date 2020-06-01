//  Copyright (c) 2012 Maciej Brodowicz
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/modules/errors.hpp>
#include <hpx/util/thread_mapper.hpp>

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <string>
#include <thread>

#if defined(__linux__) && !defined(__ANDROID__) && !defined(ANDROID)
#include <sys/syscall.h>
#elif defined(HPX_WINDOWS)
#include <windows.h>
#endif

namespace hpx { namespace util {

    namespace detail {

        ///////////////////////////////////////////////////////////////////////
        unsigned long get_system_thread_id()
        {
#if defined(__linux__) && !defined(__ANDROID__) && !defined(ANDROID)
            // this has been tested only on x86_*
            return syscall(SYS_gettid);
#elif defined(HPX_WINDOWS)
            return GetCurrentThreadId();
#else
            return invalid_tid;
#endif
        }

        // thread-specific data
        thread_data::thread_data(
            std::string const& label, basic_execution::thread_type type)
          : label_(label)
          , tid_(get_system_thread_id())
          , cleanup_()
          , type_(type)
        {
        }

        void thread_data::invalidate()
        {
            tid_ = thread_mapper::invalid_tid;
            cleanup_.reset();
        }

        bool thread_data::is_valid() const
        {
            return tid_ != thread_mapper::invalid_tid;
        }
    }    // namespace detail

    ///////////////////////////////////////////////////////////////////////////
    thread_mapper::thread_mapper() = default;

    thread_mapper::~thread_mapper()
    {
        std::lock_guard<mutex_type> m(mtx_);

        std::size_t i = 0;
        for (auto&& tinfo : thread_map_)
        {
            if (tinfo.cleanup_)
            {
                tinfo.cleanup_(i++);
            }
        }
    }

    std::uint32_t thread_mapper::register_thread(
        char const* name, basic_execution::thread_type type)
    {
        std::lock_guard<mutex_type> m(mtx_);

        auto tid = detail::get_system_thread_id();
        for (auto&& tinfo : thread_map_)
        {
            if (tinfo.tid_ == tid)
            {
                HPX_THROW_EXCEPTION(bad_parameter,
                    "thread_mapper::register_thread",
                    "thread already registered");
            }
        }

        // create mappings
        thread_map_.push_back(detail::thread_data(name, type));

        std::size_t idx = thread_map_.size() - 1;
        label_map_[name] = idx;

        return static_cast<std::uint32_t>(idx);
    }

    bool thread_mapper::unregister_thread()
    {
        std::lock_guard<mutex_type> m(mtx_);

        std::size_t i = 0;
        auto tid = detail::get_system_thread_id();
        for (auto&& tinfo : thread_map_)
        {
            if (tinfo.tid_ == tid)
            {
                label_map_.erase(tinfo.label_);
                if (tinfo.cleanup_)
                {
                    tinfo.cleanup_(i);
                }
                tinfo.invalidate();
                return true;
            }
            ++i;
        }
        return false;
    }

    bool thread_mapper::register_callback(
        std::uint32_t tix, callback_type const& cb)
    {
        std::lock_guard<mutex_type> m(mtx_);

        auto idx = static_cast<std::size_t>(tix);
        if (idx >= thread_map_.size() || !thread_map_[tix].is_valid())
        {
            return false;
        }

        thread_map_[tix].cleanup_ = cb;
        return true;
    }

    bool thread_mapper::revoke_callback(std::uint32_t tix)
    {
        std::lock_guard<mutex_type> m(mtx_);

        auto idx = static_cast<std::size_t>(tix);
        if (idx >= thread_map_.size() || !thread_map_[tix].is_valid())
        {
            return false;
        }

        thread_map_[tix].cleanup_.reset();
        return true;
    }

    unsigned long thread_mapper::get_thread_id(std::uint32_t tix) const
    {
        std::lock_guard<mutex_type> m(mtx_);

        auto idx = static_cast<std::size_t>(tix);
        if (idx >= thread_map_.size())
        {
            return thread_mapper::invalid_tid;
        }
        return thread_map_[idx].tid_;
    }

    std::string const& thread_mapper::get_thread_label(std::uint32_t tix) const
    {
        std::lock_guard<mutex_type> m(mtx_);

        auto idx = static_cast<std::size_t>(tix);
        if (idx >= thread_map_.size())
        {
            static std::string invalid_label;
            return invalid_label;
        }
        return thread_map_[idx].label_;
    }

    basic_execution::thread_type thread_mapper::get_thread_type(
        std::uint32_t tix) const
    {
        std::lock_guard<mutex_type> m(mtx_);

        auto idx = static_cast<std::size_t>(tix);
        if (idx >= thread_map_.size())
        {
            return basic_execution::thread_type::unknown;
        }
        return thread_map_[idx].type_;
    }

    std::uint32_t thread_mapper::get_thread_index(
        std::string const& label) const
    {
        std::lock_guard<mutex_type> m(mtx_);

        auto it = label_map_.find(label);
        if (it == label_map_.end())
        {
            return invalid_index;
        }
        return thread_map_[it->second].tid_;
    }

    std::uint32_t thread_mapper::get_thread_count() const
    {
        std::lock_guard<mutex_type> m(mtx_);

        return static_cast<std::uint32_t>(label_map_.size());
    }
}}    // namespace hpx::util
