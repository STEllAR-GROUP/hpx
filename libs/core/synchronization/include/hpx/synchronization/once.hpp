//  (C) Copyright 2005-7 Anthony Williams
//  (C) Copyright 2005 John Maddock
//  (C) Copyright 2011-2012 Vicente J. Botet Escriba
//  Copyright (c) 2013 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See
//  accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/functional/detail/invoke.hpp>
#include <hpx/synchronization/event.hpp>

#include <atomic>
#include <utility>

namespace hpx { namespace lcos { namespace local {
    struct once_flag
    {
    public:
        HPX_NON_COPYABLE(once_flag);

    public:
        once_flag() noexcept
          : status_(0)
        {
        }

    private:
        std::atomic<long> status_;
        lcos::local::event event_;

        template <typename F, typename... Args>
        friend void call_once(once_flag& flag, F&& f, Args&&... args);
    };

#define HPX_ONCE_INIT ::hpx::lcos::local::once_flag()

    ///////////////////////////////////////////////////////////////////////////
    template <typename F, typename... Args>
    void call_once(once_flag& flag, F&& f, Args&&... args)
    {
        // Try for a quick win: if the procedure has already been called
        // just skip through:
        long const function_complete_flag_value = 0xc15730e2;
        long const running_value = 0x7f0725e3;

        while (flag.status_.load(std::memory_order_acquire) !=
            function_complete_flag_value)
        {
            long status = 0;
            if (flag.status_.compare_exchange_strong(status, running_value))
            {
                try
                {
                    // reset event to ensure its usability in case the
                    // wrapped function was throwing an exception before
                    flag.event_.reset();

                    HPX_INVOKE(HPX_FORWARD(F, f), HPX_FORWARD(Args, args)...);

                    // set status to done, release waiting threads
                    flag.status_.store(function_complete_flag_value);
                    flag.event_.set();
                    break;
                }
                catch (...)
                {
                    // reset status to initial, release waiting threads
                    flag.status_.store(0);
                    flag.event_.set();

                    throw;
                }
            }

            // we're done if function was called
            if (status == function_complete_flag_value)
                break;

            // wait for the function finish executing
            flag.event_.wait();
        }
    }
}}}    // namespace hpx::lcos::local
