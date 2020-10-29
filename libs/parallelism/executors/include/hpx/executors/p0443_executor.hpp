//  Copyright (c) 2020 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/execution_base/executor.hpp>

namespace hpx { namespace execution { namespace experimental {
    struct p0443_executor
    {
        constexpr p0443_executor() = default;

        p0443_executor require(hpx::threads::thread_schedule_hint hint)
        {
            auto exec = *this;
            exec.schedulehint_ = hint;
            return exec;
        }

        hpx::threads::thread_schedule_hint query(
            hpx::threads::thread_schedule_hint)
        {
            return schedulehint_;
        }

        /// \cond NOINTERNAL
        bool operator==(p0443_executor const& rhs) const noexcept
        {
            return pool_ == rhs.pool_ && priority_ == rhs.priority_ &&
                stacksize_ == rhs.stacksize_ &&
                schedulehint_ == rhs.schedulehint_;
        }

        bool operator!=(p0443_executor const& rhs) const noexcept
        {
            return !(*this == rhs);
        }

        template <typename F>
        void execute(F&& f) const
        {
            hpx::util::thread_description desc(f);

            hpx::parallel::execution::detail::post_policy_dispatch<
                hpx::launch::async_policy>::call(hpx::launch::async, desc,
                pool_, priority_, stacksize_, schedulehint_,
                std::forward<F>(f));
        }

        template <typename F, typename N>
        void bulk_execute(F&& f, N n) const
        {
            hpx::util::thread_description desc(f);

            for (std::size_t i = 0; i < static_cast<std::size_t>(n); ++i)
            {
                hpx::parallel::execution::detail::post_policy_dispatch<
                    hpx::launch::async_policy>::call(hpx::launch::async, desc,
                    pool_, priority_, stacksize_, schedulehint_,
                    std::forward<F>(f), i);
            }
        }
        /// \endcond

    private:
        /// \cond NOINTERNAL
        hpx::threads::thread_pool_base* pool_ =
            hpx::threads::detail::get_self_or_default_pool();
        hpx::threads::thread_priority priority_ =
            hpx::threads::thread_priority_normal;
        hpx::threads::thread_stacksize stacksize_ =
            hpx::threads::thread_stacksize_small;
        hpx::threads::thread_schedule_hint schedulehint_{};
        /// \endcond
    };
}}}    // namespace hpx::execution::experimental

namespace hpx { namespace threads {
    template <>
    struct thread_schedule_hint::is_applicable_property<
        hpx::execution::experimental::p0443_executor> : std::true_type
    {
    };
}}    // namespace hpx::threads
