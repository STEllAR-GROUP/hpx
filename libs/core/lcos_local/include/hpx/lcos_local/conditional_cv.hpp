//  Copyright (c) 2007-2023 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/assert.hpp>
#include <hpx/functional/move_only_function.hpp>
#include <hpx/synchronization/condition_variable.hpp>

#include <utility>

namespace hpx::lcos::local {

    ///////////////////////////////////////////////////////////////////////////
    struct conditional_cv
    {
        conditional_cv() = default;
        ~conditional_cv() = default;

        conditional_cv(conditional_cv const& rhs) = delete;
        conditional_cv& operator=(conditional_cv const& rhs) = delete;
        conditional_cv(conditional_cv&& rhs) = delete;
        conditional_cv& operator=(conditional_cv&& rhs) = delete;

        // Wait for the trigger to fire
        template <typename Condition, typename Lock>
        void wait(Condition&& func, Lock& l)
        {
            cond_.assign(HPX_FORWARD(Condition, func));
            cv_.wait(l);
        }

        // Trigger this object.
        template <typename Lock>
        bool set(Lock& l)
        {
            // trigger this object
            HPX_ASSERT(cond_);
            if (cond_())
            {
                cv_.notify_all(l, threads::thread_priority::boost, false);
                return true;
            }
            return false;
        }

    private:
        detail::condition_variable cv_;
        hpx::move_only_function<bool()> cond_;
    };
}    // namespace hpx::lcos::local
