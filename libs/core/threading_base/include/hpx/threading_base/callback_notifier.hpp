//  Copyright (c) 2007-2022 Hartmut Kaiser
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
namespace hpx::threads::policies {

    class HPX_CORE_EXPORT callback_notifier
    {
    public:
        using on_startstop_type = hpx::function<void(
            std::size_t, std::size_t, char const*, char const*)>;
        using on_error_type =
            hpx::function<bool(std::size_t, std::exception_ptr const&)>;

        callback_notifier();

        void on_start_thread(std::size_t local_thread_num,
            std::size_t global_thread_num, char const* pool_name,
            char const* postfix) const;

        void on_stop_thread(std::size_t local_thread_num,
            std::size_t global_thread_num, char const* pool_name,
            char const* postfix) const;

        bool on_error(
            std::size_t global_thread_num, std::exception_ptr const& e) const;
        void add_on_start_thread_callback(on_startstop_type const& callback);
        void add_on_stop_thread_callback(on_startstop_type const& callback);
        void set_on_error_callback(on_error_type const& callback);

        // functions to call for each created thread
        std::deque<on_startstop_type> on_start_thread_callbacks_;
        // functions to call in case of unexpected stop
        std::deque<on_startstop_type> on_stop_thread_callbacks_;
        // functions to call in case of error
        on_error_type on_error_;
    };
}    // namespace hpx::threads::policies

#include <hpx/config/warnings_suffix.hpp>
