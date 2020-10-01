//  Copyright (c) 2016 Bibek Wagle
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/functional/function.hpp>
#include <hpx/timing/steady_clock.hpp>

#include <memory>
#include <string>

namespace hpx { namespace util { namespace detail
{
    class pool_timer;
}}}

namespace hpx { namespace util
{
    class HPX_EXPORT pool_timer
    {
    public:
        HPX_NON_COPYABLE(pool_timer);

    public:
        pool_timer();

        pool_timer(util::function_nonser<bool()> const& f,
            util::function_nonser<void()> const& on_term,
            std::string const& description = "",
            bool pre_shutdown = true);

        ~pool_timer();

        bool start(hpx::chrono::steady_duration const& time_duration,
            bool evaluate = false);
        bool stop();

        bool is_started() const;
        bool is_terminated() const;

    private:
        std::shared_ptr<detail::pool_timer> timer_;
    };
}}


