//  Copyright (c) 2007-2022 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#include <hpx/functional/function.hpp>
#include <hpx/threading_base/callback_notifier.hpp>

#include <cstddef>
#include <deque>
#include <exception>

///////////////////////////////////////////////////////////////////////////////
namespace hpx::threads::policies {

    callback_notifier::callback_notifier() = default;

    void callback_notifier::on_start_thread(std::size_t local_thread_num,
        std::size_t global_thread_num, char const* pool_name,
        char const* postfix) const
    {
        for (auto& callback : on_start_thread_callbacks_)
        {
            if (callback)
            {
                callback(
                    local_thread_num, global_thread_num, pool_name, postfix);
            }
        }
    }

    void callback_notifier::on_stop_thread(std::size_t local_thread_num,
        std::size_t global_thread_num, char const* pool_name,
        char const* postfix) const
    {
        for (auto& callback : on_stop_thread_callbacks_)
        {
            if (callback)
            {
                callback(
                    local_thread_num, global_thread_num, pool_name, postfix);
            }
        }
    }

    bool callback_notifier::on_error(
        std::size_t global_thread_num, std::exception_ptr const& e) const
    {
        if (on_error_)
        {
            return on_error_(global_thread_num, e);
        }
        return true;
    }

    void callback_notifier::add_on_start_thread_callback(
        on_startstop_type const& callback)
    {
        on_start_thread_callbacks_.push_back(callback);
    }

    void callback_notifier::add_on_stop_thread_callback(
        on_startstop_type const& callback)
    {
        on_stop_thread_callbacks_.push_front(callback);
    }

    void callback_notifier::set_on_error_callback(on_error_type const& callback)
    {
        on_error_ = callback;
    }
}    // namespace hpx::threads::policies
