//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/functional/function.hpp>

#include <cstddef>
#include <deque>
#include <exception>

#include <hpx/config/warnings_prefix.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace threads { namespace policies {
    class HPX_CORE_EXPORT callback_notifier
    {
    public:
        typedef util::function_nonser<void(
            std::size_t, std::size_t, char const*, char const*)>
            on_startstop_type;
        typedef util::function_nonser<bool(
            std::size_t, std::exception_ptr const&)>
            on_error_type;

        callback_notifier()
          : on_start_thread_callbacks_()
          , on_stop_thread_callbacks_()
          , on_error_()
        {
        }

        void on_start_thread(std::size_t local_thread_num,
            std::size_t global_thread_num, char const* pool_name,
            char const* postfix) const
        {
            for (auto& callback : on_start_thread_callbacks_)
            {
                if (callback)
                {
                    callback(local_thread_num, global_thread_num, pool_name,
                        postfix);
                }
            }
        }

        void on_stop_thread(std::size_t local_thread_num,
            std::size_t global_thread_num, char const* pool_name,
            char const* postfix) const
        {
            for (auto& callback : on_stop_thread_callbacks_)
            {
                if (callback)
                {
                    callback(local_thread_num, global_thread_num, pool_name,
                        postfix);
                }
            }
        }

        bool on_error(
            std::size_t global_thread_num, std::exception_ptr const& e) const
        {
            if (on_error_)
            {
                return on_error_(global_thread_num, e);
            }
            return true;
        }

        void add_on_start_thread_callback(on_startstop_type const& callback)
        {
            on_start_thread_callbacks_.push_back(callback);
        }

        void add_on_stop_thread_callback(on_startstop_type const& callback)
        {
            on_stop_thread_callbacks_.push_front(callback);
        }

        void set_on_error_callback(on_error_type const& callback)
        {
            on_error_ = callback;
        }

        // functions to call for each created thread
        std::deque<on_startstop_type> on_start_thread_callbacks_;
        // functions to call in case of unexpected stop
        std::deque<on_startstop_type> on_stop_thread_callbacks_;
        // functions to call in case of error
        on_error_type on_error_;
    };

}}}    // namespace hpx::threads::policies

#include <hpx/config/warnings_suffix.hpp>
