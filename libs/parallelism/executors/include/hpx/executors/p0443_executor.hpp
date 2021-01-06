//  Copyright (c) 2020 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/execution/detail/post_policy_dispatch.hpp>
#include <hpx/execution_base/receiver.hpp>
#include <hpx/execution_base/sender.hpp>

#include <cstddef>
#include <exception>
#include <type_traits>
#include <utility>

namespace hpx { namespace execution { namespace experimental {
    struct executor
    {
        constexpr executor() = default;

        /// \cond NOINTERNAL
        bool operator==(executor const& rhs) const noexcept
        {
            return pool_ == rhs.pool_ && priority_ == rhs.priority_ &&
                stacksize_ == rhs.stacksize_ &&
                schedulehint_ == rhs.schedulehint_;
        }

        bool operator!=(executor const& rhs) const noexcept
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

        template <typename R>
        struct operation_state
        {
            typename std::decay<R>::type r;

            void start() noexcept
            {
                try
                {
                    hpx::execution::experimental::execute(
                        executor{}, [r = std::move(r)]() mutable {
                            hpx::execution::experimental::set_value(
                                std::move(r));
                        });
                }
                catch (...)
                {
                    hpx::execution::experimental::set_error(
                        std::move(r), std::current_exception());
                }
            }
        };

        struct sender
        {
            template <template <typename...> class Tuple,
                template <typename...> class Variant>
            using value_types = Variant<Tuple<>>;

            template <template <typename...> class Variant>
            using error_types = Variant<std::exception_ptr>;

            static constexpr bool sends_done = false;

            template <typename R>
            operation_state<R> connect(R&& r)
            {
                return {std::forward<R>(r)};
            }
        };

        template <template <class...> class Tuple,
            template <class...> class Variant>
        using value_types = Variant<Tuple<>>;

        template <template <class...> class Variant>
        using error_types = Variant<std::exception_ptr>;

        static constexpr bool sends_done = false;

        template <typename R>
        operation_state<R> connect(R&& r)
        {
            return {std::forward<R>(r)};
        }

        sender schedule()
        {
            return {};
        }
        /// \endcond

    private:
        /// \cond NOINTERNAL
        hpx::threads::thread_pool_base* pool_ =
            hpx::threads::detail::get_self_or_default_pool();
        hpx::threads::thread_priority priority_ =
            hpx::threads::thread_priority::normal;
        hpx::threads::thread_stacksize stacksize_ =
            hpx::threads::thread_stacksize::small_;
        hpx::threads::thread_schedule_hint schedulehint_{};
        /// \endcond
    };
}}}    // namespace hpx::execution::experimental
