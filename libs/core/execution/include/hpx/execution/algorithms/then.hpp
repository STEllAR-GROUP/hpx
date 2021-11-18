//  Copyright (c) 2020 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/concepts/concepts.hpp>
#include <hpx/errors/try_catch_exception_ptr.hpp>
#include <hpx/execution/algorithms/detail/partial_algorithm.hpp>
#include <hpx/execution_base/completion_scheduler.hpp>
#include <hpx/execution_base/receiver.hpp>
#include <hpx/execution_base/sender.hpp>
#include <hpx/functional/detail/tag_fallback_invoke.hpp>
#include <hpx/type_support/pack.hpp>

#include <exception>
#include <type_traits>
#include <utility>

namespace hpx { namespace execution { namespace experimental {
    namespace detail {
        template <typename Receiver, typename F>
        struct then_receiver
        {
            HPX_NO_UNIQUE_ADDRESS std::decay_t<Receiver> receiver;
            HPX_NO_UNIQUE_ADDRESS std::decay_t<F> f;

            template <typename Error>
            friend void tag_invoke(
                set_error_t, then_receiver&& r, Error&& error) noexcept
            {
                hpx::execution::experimental::set_error(
                    HPX_MOVE(r.receiver), HPX_FORWARD(Error, error));
            }

            friend void tag_invoke(set_done_t, then_receiver&& r) noexcept
            {
                hpx::execution::experimental::set_done(HPX_MOVE(r.receiver));
            }

        private:
            template <typename... Ts>
            void set_value_helper(Ts&&... ts) noexcept
            {
                hpx::detail::try_catch_exception_ptr(
                    [&]() {
                        if constexpr (std::is_void_v<
                                          hpx::util::invoke_result_t<F, Ts...>>)
                        {
                        // Certain versions of GCC with optimizations fail on
                        // the move with an internal compiler error.
#if defined(HPX_GCC_VERSION) && (HPX_GCC_VERSION < 100000)
                            HPX_INVOKE(std::move(f), HPX_FORWARD(Ts, ts)...);
#else
                            HPX_INVOKE(HPX_MOVE(f), HPX_FORWARD(Ts, ts)...);
#endif
                            hpx::execution::experimental::set_value(
                                HPX_MOVE(receiver));
                        }
                        else
                        {
                        // Certain versions of GCC with optimizations fail on
                        // the move with an internal compiler error.
#if defined(HPX_GCC_VERSION) && (HPX_GCC_VERSION < 100000)
                            auto&& result = HPX_INVOKE(
                                std::move(f), HPX_FORWARD(Ts, ts)...);
#else
                            auto&& result =
                                HPX_INVOKE(HPX_MOVE(f), HPX_FORWARD(Ts, ts)...);
#endif
                            hpx::execution::experimental::set_value(
                                HPX_MOVE(receiver), HPX_MOVE(result));
                        }
                    },
                    [&](std::exception_ptr ep) {
                        hpx::execution::experimental::set_error(
                            HPX_MOVE(receiver), HPX_MOVE(ep));
                    });
            }

            template <typename... Ts,
                typename = std::enable_if_t<hpx::is_invocable_v<F, Ts...>>>
            friend void tag_invoke(
                set_value_t, then_receiver&& r, Ts&&... ts) noexcept
            {
                // GCC 7 fails with an internal compiler error unless the actual
                // body is in a helper function.
                r.set_value_helper(HPX_FORWARD(Ts, ts)...);
            }
        };

        template <typename Sender, typename F>
        struct then_sender
        {
            HPX_NO_UNIQUE_ADDRESS std::decay_t<Sender> sender;
            HPX_NO_UNIQUE_ADDRESS std::decay_t<F> f;

            template <typename Tuple>
            struct invoke_result_helper;

            template <template <typename...> class Tuple, typename... Ts>
            struct invoke_result_helper<Tuple<Ts...>>
            {
                using result_type = hpx::util::invoke_result_t<F, Ts...>;
                using type =
                    typename std::conditional<std::is_void<result_type>::value,
                        Tuple<>, Tuple<result_type>>::type;
            };

            template <template <typename...> class Tuple,
                template <typename...> class Variant>
            using value_types =
                hpx::util::detail::unique_t<hpx::util::detail::transform_t<
                    typename hpx::execution::experimental::sender_traits<
                        Sender>::template value_types<Tuple, Variant>,
                    invoke_result_helper>>;

            template <template <typename...> class Variant>
            using error_types =
                hpx::util::detail::unique_t<hpx::util::detail::prepend_t<
                    typename hpx::execution::experimental::sender_traits<
                        Sender>::template error_types<Variant>,
                    std::exception_ptr>>;

            static constexpr bool sends_done = false;

            template <typename CPO,
                // clang-format off
                HPX_CONCEPT_REQUIRES_(
                    hpx::execution::experimental::detail::is_receiver_cpo_v<CPO> &&
                    hpx::execution::experimental::detail::has_completion_scheduler_v<
                        CPO, std::decay_t<Sender>>)
                // clang-format on
                >
            friend constexpr auto tag_invoke(
                hpx::execution::experimental::get_completion_scheduler_t<CPO>,
                then_sender const& sender)
            {
                return hpx::execution::experimental::get_completion_scheduler<
                    CPO>(sender.sender);
            }

            template <typename Receiver>
            friend auto tag_invoke(
                connect_t, then_sender&& s, Receiver&& receiver)
            {
                return hpx::execution::experimental::connect(HPX_MOVE(s.sender),
                    then_receiver<Receiver, F>{
                        HPX_FORWARD(Receiver, receiver), HPX_MOVE(s.f)});
            }

            template <typename Receiver>
            friend auto tag_invoke(
                connect_t, then_sender& r, Receiver&& receiver)
            {
                return hpx::execution::experimental::connect(r.sender,
                    then_receiver<Receiver, F>{
                        HPX_FORWARD(Receiver, receiver), r.f});
            }
        };
    }    // namespace detail

    inline constexpr struct then_t final
      : hpx::functional::detail::tag_fallback<then_t>
    {
    private:
        // clang-format off
        template <typename Sender, typename F,
            HPX_CONCEPT_REQUIRES_(
                is_sender_v<Sender>
            )>
        // clang-format on
        friend constexpr HPX_FORCEINLINE auto tag_fallback_invoke(
            then_t, Sender&& sender, F&& f)
        {
            return detail::then_sender<Sender, F>{
                HPX_FORWARD(Sender, sender), HPX_FORWARD(F, f)};
        }

        template <typename F>
        friend constexpr HPX_FORCEINLINE auto tag_fallback_invoke(then_t, F&& f)
        {
            return detail::partial_algorithm<then_t, F>{HPX_FORWARD(F, f)};
        }
    } then{};
}}}    // namespace hpx::execution::experimental
