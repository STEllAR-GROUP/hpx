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

        template <typename ReceiverId, typename F>
        struct then_receiver
        {
            using Receiver = hpx::meta::type<ReceiverId>;

            struct then_receiver_data
            {
                HPX_NO_UNIQUE_ADDRESS std::decay_t<Receiver> receiver;
                HPX_NO_UNIQUE_ADDRESS std::decay_t<F> f;
            };

            struct type
            {
                using id = then_receiver;
                then_receiver_data* op;

                template <typename Error>
                friend void tag_invoke(
                    set_error_t, type&& self, Error&& error) noexcept
                {
                    hpx::execution::experimental::set_error(
                        HPX_MOVE(self.op->receiver), HPX_FORWARD(Error, error));
                }

                friend void tag_invoke(set_stopped_t, type&& self) noexcept
                {
                    hpx::execution::experimental::set_stopped(
                        HPX_MOVE(self.op->receiver));
                }

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
                                HPX_INVOKE(
                                    std::move(op->f), HPX_FORWARD(Ts, ts)...);
                                hpx::execution::experimental::set_value(
                                    HPX_MOVE(op->receiver));
                            }
                            else
                            {
                                auto&& result = HPX_INVOKE(
                                    std::move(op->f), HPX_FORWARD(Ts, ts)...);
                                hpx::execution::experimental::set_value(
                                    HPX_MOVE(op->receiver), HPX_MOVE(result));
                            }
                        },
                        [&](std::exception_ptr ep) {
                            hpx::execution::experimental::set_error(
                                HPX_MOVE(op->receiver), HPX_MOVE(ep));
                        });
                }

                // clang-format off
                template <typename... Ts,
                    HPX_CONCEPT_REQUIRES_(
                        hpx::is_invocable_v<F, Ts...>
                    )>
                // clang-format on
                friend void tag_invoke(
                    set_value_t, type&& self, Ts&&... ts) noexcept
                {
                    // GCC 7 fails with an internal compiler error unless the actual
                    // body is in a helper function.
                    HPX_MOVE(self).set_value_helper(HPX_FORWARD(Ts, ts)...);
                }

                friend auto tag_invoke(get_env_t, const type& self)
                    -> hpx::util::invoke_result_t<get_env_t, const Receiver&>
                {
                    return get_env(self.op->receiver);
                }
            };
        };
#if defined(HPX_GCC_VERSION) && HPX_GCC_VERSION >= 110000
#pragma GCC diagnostic pop
#endif

        template <typename Sender, typename Receiver>
        inline constexpr bool is_nothrow_connectable_v = noexcept(
            connect(std::declval<Sender>(), std::declval<Receiver>()));

        template <typename Sender, typename ReceiverId, typename Fun>
        struct then_operation
        {
            using Receiver = hpx::meta::type<ReceiverId>;
            using ReceiverId_t = then_receiver<ReceiverId, Fun>;
            using Receiver_t = hpx::meta::type<ReceiverId_t>;

            struct type
            {
                type() = default;

                using id = then_operation;
                typename ReceiverId_t::then_receiver_data data;
                connect_result_t<Sender, Receiver_t> operation_state;

                type(Sender&& sndr, Receiver rcvr, Fun fun)    //
                    noexcept(
                        hpx::meta::is_nothrow_decay_copyable_v<Receiver>      //
                            && hpx::meta::is_nothrow_decay_copyable_v<Fun>    //
                                && is_nothrow_connectable_v<Sender, Receiver_t>)
                  : data{(Receiver &&) rcvr, (Fun &&) fun}
                  , operation_state(
                        connect((Sender &&) sndr, Receiver_t{&data}))
                {
                }

                friend void tag_invoke(start_t, type& self) noexcept
                {
                    start(self.operation_state);
                }

            private:
                type(type&&) = delete;
            };
        };

        template <typename SenderId, typename F>
        struct then_sender
        {
            using Sender = hpx::meta::type<SenderId>;
            template <typename Self, typename Receiver>
            using operation = hpx::meta::type<
                then_operation<hpx::meta::copy_cvref_t<Self, Sender>,
                    hpx::meta::get_id_t<Receiver>, F>>;

            struct type
            {
                using is_sender = void;
                using id = then_sender;
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
                friend auto tag_invoke(get_completion_signatures_t, type const&,
                    Env) -> generate_completion_signatures<Env>;

                // clang-format off
                template <typename CPO,
                    HPX_CONCEPT_REQUIRES_(
                        meta::value<meta::one_of<
                            std::decay_t<CPO>, set_value_t, set_stopped_t>>&&
                            detail::has_completion_scheduler_v<CPO, Sender>
                    )>
                // clang-format on
                friend constexpr auto tag_invoke(
                    hpx::execution::experimental::get_completion_scheduler_t<
                        CPO>
                        tag,
                    type const& s)
                {
                    return tag(s.sender);
                }

                // TODO: add forwarding_sender_query

                template <typename Self, typename Receiver,
                    typename = std::enable_if_t<
                        std::is_same_v<std::decay_t<Self>, type>>>
                friend auto
                tag_invoke(connect_t, Self&& s, Receiver&& receiver) noexcept(
                    hpx::meta::is_nothrow_constructible_from_v<
                        operation<Self, Receiver>,
                        hpx::meta::copy_cvref_t<Self, Sender>, Receiver&&,
                        hpx::meta::copy_cvref_t<Self, F>>)
                    -> operation<type, Receiver>
                {
                    return {HPX_FORWARD(Self, s).sender,
                        HPX_FORWARD(Receiver, receiver),
                        HPX_FORWARD(Self, s).f};
                }

                friend auto tag_invoke(get_env_t, const type& self)    //
                    noexcept(
                        hpx::is_nothrow_invocable_v<get_env_t, const Sender&>)
                        -> hpx::util::invoke_result_t<get_env_t, const Sender&>
                {
                    return get_env(self.sender);
                }
            };
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
        template <typename Sender, typename Fun>
        using Sender_t = hpx::meta::type<detail::then_sender<
            hpx::meta::get_id_t<std::decay_t<Sender>>, Fun>>;

        // clang-format off
        template <typename Sender, typename F,
            HPX_CONCEPT_REQUIRES_(
                is_sender_v<Sender_t<Sender,F>> &&
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
                is_sender_v<Sender_t<Sender,F>>
            )>
        // clang-format on
        friend constexpr HPX_FORCEINLINE auto tag_fallback_invoke(
            then_t, Sender&& sender, F&& f) -> Sender_t<Sender, F>
        {
            return Sender_t<Sender, F>{
                HPX_FORWARD(Sender, sender), HPX_FORWARD(F, f)};
        }

        template <typename F>
        friend constexpr HPX_FORCEINLINE auto tag_fallback_invoke(then_t, F&& f)
        {
            return detail::partial_algorithm<then_t, F>{HPX_FORWARD(F, f)};
        }
    } then{};
}    // namespace hpx::execution::experimental
