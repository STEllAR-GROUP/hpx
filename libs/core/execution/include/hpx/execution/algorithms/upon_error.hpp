//  Copyright (c) 2021 ETH Zurich
//  Copyright (c) 2022-2025 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#if defined(HPX_HAVE_STDEXEC)
#include <hpx/modules/execution_base.hpp>
#else

#include <hpx/execution/algorithms/detail/partial_algorithm.hpp>
#include <hpx/modules/concepts.hpp>
#include <hpx/modules/errors.hpp>
#include <hpx/modules/execution_base.hpp>
#include <hpx/modules/functional.hpp>
#include <hpx/modules/tag_invoke.hpp>
#include <hpx/modules/type_support.hpp>

#include <exception>
#include <type_traits>
#include <utility>

namespace hpx::execution::experimental {

    namespace detail {

#if defined(HPX_GCC_VERSION) && HPX_GCC_VERSION >= 110000
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Warray-bounds"
#endif
        // Receiver adaptor that intercepts set_error and converts it to
        // set_value by invoking f(error). set_value and set_stopped are
        // forwarded unchanged to the downstream receiver.
        HPX_CXX_CORE_EXPORT template <typename Receiver, typename F>
        struct upon_error_receiver
        {
            HPX_NO_UNIQUE_ADDRESS std::decay_t<Receiver> receiver;
            HPX_NO_UNIQUE_ADDRESS std::decay_t<F> f;

            // set_value: pass through unchanged
            template <typename... Ts>
            friend void tag_invoke(
                set_value_t, upon_error_receiver&& r, Ts&&... ts) noexcept
            {
                hpx::execution::experimental::set_value(
                    HPX_MOVE(r.receiver), HPX_FORWARD(Ts, ts)...);
            }

            // set_stopped: pass through unchanged
            friend void tag_invoke(
                set_stopped_t, upon_error_receiver&& r) noexcept
            {
                hpx::execution::experimental::set_stopped(HPX_MOVE(r.receiver));
            }

        private:
            template <typename Error>
            void set_error_helper(Error&& error) && noexcept
            {
                using result_type =
                    hpx::util::invoke_result_t<F, std::decay_t<Error>>;
                hpx::detail::try_catch_exception_ptr(
                    [&]() {
                        if constexpr (std::is_void_v<result_type>)
                        {
                            HPX_INVOKE(std::move(f), HPX_FORWARD(Error, error));
                            hpx::execution::experimental::set_value(
                                HPX_MOVE(receiver));
                        }
                        else
                        {
                            auto&& result = HPX_INVOKE(
                                std::move(f), HPX_FORWARD(Error, error));
                            hpx::execution::experimental::set_value(
                                HPX_MOVE(receiver), HPX_MOVE(result));
                        }
                    },
                    [&](std::exception_ptr ep) {
                        hpx::execution::experimental::set_error(
                            HPX_MOVE(receiver), HPX_MOVE(ep));
                    });
            }

        public:
            // clang-format off
            template <typename Error,
                HPX_CONCEPT_REQUIRES_(
                    hpx::is_invocable_v<F, std::decay_t<Error>>
                )>
            // clang-format on
            friend void tag_invoke(
                set_error_t, upon_error_receiver&& r, Error&& error) noexcept
            {
                HPX_MOVE(r).set_error_helper(HPX_FORWARD(Error, error));
            }
        };
#if defined(HPX_GCC_VERSION) && HPX_GCC_VERSION >= 110000
#pragma GCC diagnostic pop
#endif

        HPX_CXX_CORE_EXPORT template <typename Sender, typename F>
        struct upon_error_sender
        {
            using is_sender = void;

            HPX_NO_UNIQUE_ADDRESS std::decay_t<Sender> sender;
            HPX_NO_UNIQUE_ADDRESS std::decay_t<F> f;

            // For each error type E in upstream error_types, upon_error
            // converts it to a value by calling f(E), producing either
            // Tuple<> (if f returns void) or Tuple<invoke_result_t<F, E>>.
            //
            // Note: gen_value_from_error uses hpx::tuple directly since
            // transform_t requires a single-argument transformer, and in
            // practice value_types checks always use hpx::tuple.
            template <typename Error>
            struct gen_value_from_error
            {
                using result_type =
                    hpx::util::invoke_result_t<F, std::decay_t<Error>>;
                using type = std::conditional_t<std::is_void_v<result_type>,
                    hpx::tuple<>, hpx::tuple<result_type>>;
            };

            template <typename Env>
            struct generate_completion_signatures
            {
                // value_types:
                //   - upstream value completions (forwarded unchanged)
                //   - for each error type E, Tuple<invoke_result_t<F, E>>
                //     (or Tuple<> if void) added as a new value alternative
                template <template <typename...> typename Tuple,
                    template <typename...> typename Variant>
                using value_types = hpx::util::detail::unique_concat_t<
                    value_types_of_t<Sender, Env, Tuple, Variant>,
                    hpx::util::detail::transform_t<
                        error_types_of_t<Sender, Env, Variant>,
                        gen_value_from_error>>;

                // error_types: only std::exception_ptr (if f itself throws)
                template <template <typename...> typename Variant>
                using error_types = Variant<std::exception_ptr>;

                // sends_stopped: same as upstream (stopped channel untouched)
                static constexpr bool sends_stopped =
                    sends_stopped_of_v<Sender, Env>;
            };

            template <typename Env>
            friend auto tag_invoke(get_completion_signatures_t,
                upon_error_sender const&,
                Env) -> generate_completion_signatures<Env>;

            // Propagate completion scheduler for set_value_t and set_stopped_t
            // channels (error channel is consumed, not forwarded).
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
                upon_error_sender const& s)
            {
                return tag(s.sender);
            }

            template <typename Receiver>
            friend auto tag_invoke(
                connect_t, upon_error_sender&& s, Receiver&& receiver)
            {
                return hpx::execution::experimental::connect(HPX_MOVE(s.sender),
                    upon_error_receiver<Receiver, F>{
                        HPX_FORWARD(Receiver, receiver), HPX_MOVE(s.f)});
            }

            template <typename Receiver>
            friend auto tag_invoke(
                connect_t, upon_error_sender& s, Receiver&& receiver)
            {
                return hpx::execution::experimental::connect(s.sender,
                    upon_error_receiver<Receiver, F>{
                        HPX_FORWARD(Receiver, receiver), s.f});
            }
        };
    }    // namespace detail

    // execution::upon_error is the error-channel counterpart of then.
    // It attaches an invocable as a handler for the error completion of the
    // input sender. The handler is called with the error value and its return
    // value is sent as a value completion to downstream. If the handler itself
    // throws, the exception is forwarded as a set_error(exception_ptr).
    //
    // Value and stopped completions from the input sender are forwarded
    // unchanged.
    //
    // upon_error is guaranteed to not begin executing the handler before the
    // returned sender is started.
    HPX_CXX_CORE_EXPORT inline constexpr struct upon_error_t final
      : hpx::functional::detail::tag_priority<upon_error_t>
    {
    private:
        // clang-format off
        template <typename Sender, typename F,
            HPX_CONCEPT_REQUIRES_(
                is_sender_v<Sender> &&
                experimental::detail::is_completion_scheduler_tag_invocable_v<
                    hpx::execution::experimental::set_value_t,
                    Sender, upon_error_t, F
                >
            )>
        // clang-format on
        friend constexpr HPX_FORCEINLINE auto tag_override_invoke(
            upon_error_t, Sender&& sender, F&& f)
        {
            auto scheduler =
                hpx::execution::experimental::get_completion_scheduler<
                    hpx::execution::experimental::set_value_t>(sender);

            return hpx::functional::tag_invoke(upon_error_t{},
                HPX_MOVE(scheduler), HPX_FORWARD(Sender, sender),
                HPX_FORWARD(F, f));
        }

        // clang-format off
        template <typename Sender, typename F,
            HPX_CONCEPT_REQUIRES_(
                is_sender_v<Sender>
            )>
        // clang-format on
        friend constexpr HPX_FORCEINLINE auto tag_fallback_invoke(
            upon_error_t, Sender&& sender, F&& f)
        {
            return detail::upon_error_sender<Sender, F>{
                HPX_FORWARD(Sender, sender), HPX_FORWARD(F, f)};
        }

        template <typename F>
        friend constexpr HPX_FORCEINLINE auto tag_fallback_invoke(
            upon_error_t, F&& f)
        {
            return detail::partial_algorithm<upon_error_t, F>{
                HPX_FORWARD(F, f)};
        }
    } upon_error{};
}    // namespace hpx::execution::experimental

#endif
