//  Copyright (c) 2020 ETH Zurich
//  Copyright (c) 2022 Hartmut Kaiser
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
#include <hpx/execution_base/completion_signatures.hpp>
#include <hpx/execution_base/receiver.hpp>
#include <hpx/execution_base/sender.hpp>
#include <hpx/functional/detail/tag_priority_invoke.hpp>
#include <hpx/futures/future.hpp>
#include <hpx/type_support/meta.hpp>
#include <hpx/type_support/pack.hpp>

#include <exception>
#include <type_traits>
#include <utility>

namespace hpx::execution::experimental {

    namespace detail {

#if defined(HPX_GCC_VERSION) && HPX_GCC_VERSION >= 110000
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Warray-bounds"
#endif
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

            friend void tag_invoke(set_stopped_t, then_receiver&& r) noexcept
            {
                hpx::execution::experimental::set_stopped(HPX_MOVE(r.receiver));
            }

        private:
            template <typename... Ts>
            void set_value_helper(Ts&&... ts) && noexcept
            {
                using result_type = hpx::util::invoke_result_t<F, Ts...>;
                hpx::detail::try_catch_exception_ptr(
                    [&]() {
                        // Certain versions of GCC with optimizations fail on
                        // the HPX_MOVE with an internal compiler error.
                        if constexpr (std::is_void_v<result_type>)
                        {
                            HPX_INVOKE(std::move(f), HPX_FORWARD(Ts, ts)...);
                            hpx::execution::experimental::set_value(
                                HPX_MOVE(receiver));
                        }
                        else
                        {
                            auto&& result = HPX_INVOKE(
                                std::move(f), HPX_FORWARD(Ts, ts)...);
                            hpx::execution::experimental::set_value(
                                HPX_MOVE(receiver), HPX_MOVE(result));
                        }
                    },
                    [&](std::exception_ptr ep) {
                        hpx::execution::experimental::set_error(
                            HPX_MOVE(receiver), HPX_MOVE(ep));
                    });
            }

            // clang-format off
            template <typename... Ts,
                HPX_CONCEPT_REQUIRES_(
                    hpx::is_invocable_v<F, Ts...>
                )>
            // clang-format on
            friend void tag_invoke(
                set_value_t, then_receiver&& r, Ts&&... ts) noexcept
            {
                // GCC 7 fails with an internal compiler error unless the actual
                // body is in a helper function.
                HPX_MOVE(r).set_value_helper(HPX_FORWARD(Ts, ts)...);
            }
        };
#if defined(HPX_GCC_VERSION) && HPX_GCC_VERSION >= 110000
#pragma GCC diagnostic pop
#endif

        template <typename Sender, typename F>
        struct then_sender
        {
            using is_sender = void;

            HPX_NO_UNIQUE_ADDRESS std::decay_t<Sender> sender;
            HPX_NO_UNIQUE_ADDRESS std::decay_t<F> f;

            template <typename Func, typename Pack>
            struct undefined_set_value_signature;

            template <typename Pack, typename Enable = void>
            struct generate_set_value_signature
              : undefined_set_value_signature<F, Pack>
            {
            };

            template <template <typename...> typename Pack, typename... Ts>
            struct generate_set_value_signature<Pack<Ts...>,
                std::enable_if_t<hpx::is_invocable_v<F, Ts...>>>
            {
                using result_type = hpx::util::invoke_result_t<F, Ts...>;
                using type = std::conditional_t<std::is_void_v<result_type>,
                    Pack<>, Pack<result_type>>;
            };

            template <typename Pack>
            using gen_value_signature = generate_set_value_signature<Pack>;

            template <typename Env>
            struct generate_completion_signatures
            {
                template <template <typename...> typename Tuple,
                    template <typename...> typename Variant>
                using value_types = hpx::util::detail::concat_inner_packs_t<
                    hpx::util::detail::transform_t<
                        value_types_of_t<Sender, Env, Tuple, Variant>,
                        gen_value_signature>>;

                template <template <typename...> typename Variant>
                using error_types = hpx::util::detail::unique_concat_t<
                    error_types_of_t<Sender, Env, Variant>,
                    Variant<std::exception_ptr>>;

                static constexpr bool sends_stopped =
                    sends_stopped_of_v<Sender, Env>;
            };

            template <typename Env>
            friend auto tag_invoke(get_completion_signatures_t,
                then_sender const&, Env) -> generate_completion_signatures<Env>;

            // clang-format off
            template <typename CPO,
                HPX_CONCEPT_REQUIRES_(
                    meta::value<meta::one_of<
                        std::decay_t<CPO>, set_value_t, set_stopped_t>> &&
                    detail::has_completion_scheduler_v<CPO, Sender>
                )>
            // clang-format on
            friend constexpr auto tag_invoke(
                hpx::execution::experimental::get_completion_scheduler_t<CPO>
                    tag,
                then_sender const& s)
            {
                return tag(s.sender);
            }

            // TODO: add forwarding_sender_query

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
                connect_t, then_sender& s, Receiver&& receiver)
            {
                return hpx::execution::experimental::connect(s.sender,
                    then_receiver<Receiver, F>{
                        HPX_FORWARD(Receiver, receiver), s.f});
            }
        };
    }    // namespace detail

    // execution::then is used to attach an invocable as a continuation for the
    // successful completion of the input sender.
    //
    // then returns a sender describing the task graph described by the input
    // sender, with an added node of invoking the provided function with the
    // values sent by the input sender as arguments.
    //
    // execution::then is guaranteed to not begin executing the function before
    // the returned sender is started.
    inline constexpr struct then_t final
      : hpx::functional::detail::tag_priority<then_t>
    {
    private:
        // clang-format off
        template <typename Sender, typename F,
            HPX_CONCEPT_REQUIRES_(
                is_sender_v<Sender> &&
                experimental::detail::is_completion_scheduler_tag_invocable_v<
                    hpx::execution::experimental::set_value_t,
                    Sender, then_t, F
                >
            )>
        // clang-format on
        friend constexpr HPX_FORCEINLINE auto tag_override_invoke(
            then_t, Sender&& sender, F&& f)
        {
            auto scheduler =
                hpx::execution::experimental::get_completion_scheduler<
                    hpx::execution::experimental::set_value_t>(sender);

            return hpx::functional::tag_invoke(then_t{}, HPX_MOVE(scheduler),
                HPX_FORWARD(Sender, sender), HPX_FORWARD(F, f));
        }

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
}    // namespace hpx::execution::experimental
