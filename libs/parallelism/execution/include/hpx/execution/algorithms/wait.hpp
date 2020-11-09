//  Copyright (c) 2020 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/execution_base/operation_state.hpp>
#include <hpx/execution_base/sender.hpp>
#include <hpx/synchronization/condition_variable.hpp>
#include <hpx/synchronization/mutex.hpp>

namespace hpx { namespace execution { namespace experimental {
    namespace detail {
        struct wait_receiver
        {
            struct state
            {
                hpx::lcos::local::condition_variable cv;
                hpx::lcos::local::mutex m;
                bool done = false;
            };

            state& st;

            void signal_done() noexcept
            {
                std::unique_lock<hpx::lcos::local::mutex> l(st.m);
                st.done = true;
                st.cv.notify_one();
            }

            void set_error(std::exception_ptr ep) noexcept
            {
                signal_done();
            }

            void set_done() noexcept
            {
                signal_done();
            };

            template <typename... Ts>
            void set_value(Ts&&... ts)
            {
                signal_done();
            }
        };
    }    // namespace detail

    // Variant of sync_wait, which does not return the value from the
    // predecessor sender. Avoids having to store the return value. TODO: Only
    // here for symmetry with future::wait. Do we want it?
    template <typename S>
    void wait(S&& s)
    {
        using state_type = detail::wait_receiver::state;

        state_type st{};
        hpx::execution::experimental::start(
            hpx::execution::experimental::connect(
                std::forward<S>(s), detail::wait_receiver{st}));

        {
            std::unique_lock<hpx::lcos::local::mutex> l(st.m);
            if (!st.done)
            {
                st.cv.wait(l);
            }
        }
    }
}}}    // namespace hpx::execution::experimental
