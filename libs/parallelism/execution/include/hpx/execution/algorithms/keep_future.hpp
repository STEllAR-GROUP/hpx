//  Copyright (c) 2021 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/assert.hpp>
#include <hpx/execution/algorithms/detail/partial_algorithm.hpp>
#include <hpx/execution_base/operation_state.hpp>
#include <hpx/execution_base/receiver.hpp>
#include <hpx/execution_base/sender.hpp>
#include <hpx/functional/tag_fallback_dispatch.hpp>
#include <hpx/futures/detail/future_data.hpp>
#include <hpx/futures/future.hpp>
#include <hpx/futures/traits/acquire_shared_state.hpp>

#include <exception>
#include <type_traits>
#include <utility>

namespace hpx { namespace execution { namespace experimental {
    namespace detail {
        template <typename Receiver, typename Future>
        struct operation_state
        {
            std::decay_t<Receiver> receiver;
            std::decay_t<Future> future;

            void start() & noexcept
            {
                try
                {
                    auto state = hpx::traits::detail::get_shared_state(future);

                    if (!state)
                    {
                        HPX_THROW_EXCEPTION(no_state, "operation_state::start",
                            "the future has no valid shared state asdasd");
                    }

                    state->set_on_completed(
                        [this, receiver = std::move(receiver)]() mutable {
                            hpx::execution::experimental::set_value(
                                std::move(receiver), std::move(future));
                        });
                }
                catch (...)
                {
                    hpx::execution::experimental::set_error(
                        std::move(receiver), std::current_exception());
                }
            }
        };
        template <typename Future>
        struct keep_future_sender_base
        {
            std::decay_t<Future> future;

            template <template <typename...> class Tuple,
                template <typename...> class Variant>
            using value_types = Variant<Tuple<std::decay_t<Future>>>;

            template <template <typename...> class Variant>
            using error_types = Variant<std::exception_ptr>;

            static constexpr bool sends_done = false;
        };

        template <typename Future>
        struct keep_future_sender;

        template <typename T>
        struct keep_future_sender<hpx::future<T>>
          : public keep_future_sender_base<hpx::future<T>>
        {
            using future_type = hpx::future<T>;
            using base_type = keep_future_sender_base<hpx::future<T>>;
            using base_type::future;

            template <typename Future,
                typename = std::enable_if_t<!std::is_same<std::decay_t<Future>,
                    keep_future_sender>::value>>
            explicit keep_future_sender(Future&& future)
              : base_type{std::forward<Future>(future)}
            {
            }

            keep_future_sender(keep_future_sender&&) = default;
            keep_future_sender& operator=(keep_future_sender&&) = default;
            keep_future_sender(keep_future_sender const&) = delete;
            keep_future_sender& operator=(keep_future_sender const&) = delete;

            template <typename Receiver>
            operation_state<Receiver, future_type> connect(
                Receiver&& receiver) &&
            {
                return {std::forward<Receiver>(receiver), std::move(future)};
            }
        };

        template <typename T>
        struct keep_future_sender<hpx::shared_future<T>>
          : keep_future_sender_base<hpx::shared_future<T>>
        {
            using future_type = hpx::shared_future<T>;
            using base_type = keep_future_sender_base<hpx::shared_future<T>>;
            using base_type::future;

            template <typename Future,
                typename = std::enable_if_t<!std::is_same<std::decay_t<Future>,
                    keep_future_sender>::value>>
            explicit keep_future_sender(Future&& future)
              : base_type{std::forward<Future>(future)}
            {
            }

            keep_future_sender(keep_future_sender&&) = default;
            keep_future_sender& operator=(keep_future_sender&&) = default;
            keep_future_sender(keep_future_sender const&) = default;
            keep_future_sender& operator=(keep_future_sender const&) = default;

            template <typename Receiver>
            operation_state<Receiver, future_type> connect(
                Receiver&& receiver) &&
            {
                return {std::forward<Receiver>(receiver), std::move(future)};
            }

            template <typename Receiver>
            operation_state<Receiver, future_type> connect(
                Receiver&& receiver) &
            {
                return {std::forward<Receiver>(receiver), future};
            }
        };
    }    // namespace detail

    HPX_INLINE_CONSTEXPR_VARIABLE struct keep_future_t final
      : hpx::functional::tag_fallback<keep_future_t>
    {
    private:
        template <typename Future,
            typename = std::enable_if_t<
                hpx::traits::is_future<std::decay_t<Future>>::value>>
        friend constexpr HPX_FORCEINLINE auto tag_fallback_dispatch(
            keep_future_t, Future&& future)
        {
            return detail::keep_future_sender<std::decay_t<Future>>(
                std::forward<Future>(future));
        }

        friend constexpr HPX_FORCEINLINE auto tag_fallback_dispatch(keep_future_t)
        {
            return detail::partial_algorithm<keep_future_t>{};
        }
    } keep_future{};
}}}    // namespace hpx::execution::experimental
