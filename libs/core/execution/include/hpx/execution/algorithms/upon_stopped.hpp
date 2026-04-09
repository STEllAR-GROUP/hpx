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
        // Receiver adaptor that intercepts set_stopped and converts it to
        // set_value by invoking f(). set_value and set_error are forwarded
        // unchanged to the downstream receiver.
        HPX_CXX_CORE_EXPORT template <typename Receiver, typename F>
        struct upon_stopped_receiver
        {
            HPX_NO_UNIQUE_ADDRESS std::decay_t<Receiver> receiver;
            HPX_NO_UNIQUE_ADDRESS std::decay_t<F> f;

            // set_value: pass through unchanged
            template <typename... Ts>
            friend void tag_invoke(
                set_value_t, upon_stopped_receiver&& r, Ts&&... ts) noexcept
            {
                hpx::execution::experimental::set_value(
                    HPX_MOVE(r.receiver), HPX_FORWARD(Ts, ts)...);
            }

            // set_error: pass through unchanged
            template <typename Error>
            friend void tag_invoke(
                set_error_t, upon_stopped_receiver&& r, Error&& error) noexcept
            {
                hpx::execution::experimental::set_error(
                    HPX_MOVE(r.receiver), HPX_FORWARD(Error, error));
            }

        private:
            void set_stopped_helper() && noexcept
            {
                using result_type = hpx::util::invoke_result_t<F>;
                hpx::detail::try_catch_exception_ptr(
                    [&]() {
                        if constexpr (std::is_void_v<result_type>)
                        {
                            HPX_INVOKE(std::move(f), );
                            hpx::execution::experimental::set_value(
                                HPX_MOVE(receiver));
                        }
                        else
                        {
                            auto&& result = HPX_INVOKE(std::move(f), );
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
            friend void tag_invoke(
                set_stopped_t, upon_stopped_receiver&& r) noexcept
            {
                HPX_MOVE(r).set_stopped_helper();
            }
        };
#if defined(HPX_GCC_VERSION) && HPX_GCC_VERSION >= 110000
#pragma GCC diagnostic pop
#endif

        HPX_CXX_CORE_EXPORT template <typename Sender, typename F>
        struct upon_stopped_sender
        {
            using is_sender = void;

            HPX_NO_UNIQUE_ADDRESS std::decay_t<Sender> sender;
            HPX_NO_UNIQUE_ADDRESS std::decay_t<F> f;

            template <typename Env>
            struct generate_completion_signatures
            {
            private:
                // The value type produced when the stopped signal is handled:
                // either Tuple<> if f() returns void, or Tuple<result_type>.
                // Uses Tuple template parameter to stay consistent with
                // value_types_of_t.
                using stopped_result_type = hpx::util::invoke_result_t<F>;

                template <template <typename...> typename Tuple>
                using stopped_value_tuple =
                    std::conditional_t<std::is_void_v<stopped_result_type>,
                        Tuple<>, Tuple<stopped_result_type>>;

            public:
                // value_types:
                //   - upstream value completions (forwarded unchanged)
                //   - stopped_value_tuple<Tuple> added as a new alternative,
                //     representing the value produced by the stopped handler
                template <template <typename...> typename Tuple,
                    template <typename...> typename Variant>
                using value_types = hpx::util::detail::unique_concat_t<
                    value_types_of_t<Sender, Env, Tuple, Variant>,
                    Variant<stopped_value_tuple<Tuple>>>;

                // error_types: upstream errors + std::exception_ptr (if f throws)
                template <template <typename...> typename Variant>
                using error_types = hpx::util::detail::unique_concat_t<
                    error_types_of_t<Sender, Env, Variant>,
                    Variant<std::exception_ptr>>;

                // sends_stopped: false — the stopped signal is consumed by f
                static constexpr bool sends_stopped = false;
            };

            template <typename Env>
            friend auto tag_invoke(get_completion_signatures_t,
                upon_stopped_sender const&,
                Env) -> generate_completion_signatures<Env>;

            // Propagate completion scheduler for set_value_t channel.
            // set_error_t is forwarded, set_stopped_t is consumed.
            // clang-format off
            template <typename CPO,
                HPX_CONCEPT_REQUIRES_(
                    std::is_same_v<std::decay_t<CPO>, set_value_t> &&
                    detail::has_completion_scheduler_v<CPO, Sender>
                )>
            // clang-format on
            friend constexpr auto tag_invoke(
                hpx::execution::experimental::get_completion_scheduler_t<CPO>
                    tag,
                upon_stopped_sender const& s)
            {
                return tag(s.sender);
            }

            template <typename Receiver>
            friend auto tag_invoke(
                connect_t, upon_stopped_sender&& s, Receiver&& receiver)
            {
                return hpx::execution::experimental::connect(HPX_MOVE(s.sender),
                    upon_stopped_receiver<Receiver, F>{
                        HPX_FORWARD(Receiver, receiver), HPX_MOVE(s.f)});
            }

            template <typename Receiver>
            friend auto tag_invoke(
                connect_t, upon_stopped_sender& s, Receiver&& receiver)
            {
                return hpx::execution::experimental::connect(s.sender,
                    upon_stopped_receiver<Receiver, F>{
                        HPX_FORWARD(Receiver, receiver), s.f});
            }
        };
    }    // namespace detail

    // execution::upon_stopped is the stopped-channel counterpart of then.
    // It attaches an invocable as a handler for the stopped completion of the
    // input sender. The handler is called with no arguments and its return
    // value is sent as a value completion to downstream. If the handler itself
    // throws, the exception is forwarded as a set_error(exception_ptr).
    //
    // Value and error completions from the input sender are forwarded
    // unchanged. The returned sender never sends a stopped signal.
    //
    // upon_stopped is guaranteed to not begin executing the handler before the
    // returned sender is started.
    HPX_CXX_CORE_EXPORT inline constexpr struct upon_stopped_t final
      : hpx::functional::detail::tag_priority<upon_stopped_t>
    {
    private:
        // clang-format off
        template <typename Sender, typename F,
            HPX_CONCEPT_REQUIRES_(
                is_sender_v<Sender> &&
                experimental::detail::is_completion_scheduler_tag_invocable_v<
                    hpx::execution::experimental::set_value_t,
                    Sender, upon_stopped_t, F
                >
            )>
        // clang-format on
        friend constexpr HPX_FORCEINLINE auto tag_override_invoke(
            upon_stopped_t, Sender&& sender, F&& f)
        {
            auto scheduler =
                hpx::execution::experimental::get_completion_scheduler<
                    hpx::execution::experimental::set_value_t>(sender);

            return hpx::functional::tag_invoke(upon_stopped_t{},
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
            upon_stopped_t, Sender&& sender, F&& f)
        {
            return detail::upon_stopped_sender<Sender, F>{
                HPX_FORWARD(Sender, sender), HPX_FORWARD(F, f)};
        }

        template <typename F>
        friend constexpr HPX_FORCEINLINE auto tag_fallback_invoke(
            upon_stopped_t, F&& f)
        {
            return detail::partial_algorithm<upon_stopped_t, F>{
                HPX_FORWARD(F, f)};
        }
    } upon_stopped{};
}    // namespace hpx::execution::experimental

#endif
