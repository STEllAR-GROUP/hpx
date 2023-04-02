//  Copyright (c) 2020 ETH Zurich
//  Copyright (c) 2022 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/async_base/dataflow.hpp>
#include <hpx/async_base/launch_policy.hpp>
#include <hpx/concepts/concepts.hpp>
#include <hpx/datastructures/optional.hpp>
#include <hpx/datastructures/variant.hpp>
#include <hpx/execution/algorithms/detail/single_result.hpp>
#include <hpx/execution/algorithms/transfer.hpp>
#include <hpx/execution/queries/get_stop_token.hpp>
#include <hpx/execution_base/completion_signatures.hpp>
#include <hpx/execution_base/get_env.hpp>
#include <hpx/execution_base/operation_state.hpp>
#include <hpx/execution_base/receiver.hpp>
#include <hpx/execution_base/sender.hpp>
#include <hpx/functional/detail/tag_fallback_invoke.hpp>
#include <hpx/functional/invoke_fused.hpp>
#include <hpx/synchronization/stop_token.hpp>
#include <hpx/type_support/meta.hpp>
#include <hpx/type_support/pack.hpp>

#include <atomic>
#include <cstddef>
#include <exception>
#include <functional>
#include <memory>
#include <type_traits>
#include <utility>

namespace hpx::execution::experimental {
    namespace detail {

        // callback object to request cancellation
        struct on_stop_requested
        {
            hpx::experimental::in_place_stop_source& stop_source_;
            void operator()() noexcept
            {
                stop_source_.request_stop();
            }
        };

        // This is a receiver to be connected to the ith predecessor sender
        // passed to when_all. When set_value is called, it will emplace the
        // values sent into the appropriate position in the pack used to store
        // values from all predecessor senders.
        template <typename OperationState>
        struct when_all_receiver
        {
            std::decay_t<OperationState>& op_state;

            explicit when_all_receiver(
                std::decay_t<OperationState>& op_state) noexcept
              : op_state(op_state)
            {
            }

            template <typename Error>
            friend void tag_invoke(
                set_error_t, when_all_receiver&& r, Error&& error) noexcept
            {
                if (!r.op_state.set_stopped_error_called.exchange(true))
                {
                    r.op_state.stop_source_.request_stop();
                    try
                    {
                        r.op_state.error = HPX_FORWARD(Error, error);
                    }
                    catch (...)
                    {
                        // NOLINTNEXTLINE(bugprone-throw-keyword-missing)
                        r.op_state.error = std::current_exception();
                    }
                }
                r.op_state.finish();
            }

            friend void tag_invoke(
                set_stopped_t, when_all_receiver&& r) noexcept
            {
                // request stop only if we're not in error state
                if (!r.op_state.set_stopped_error_called.exchange(true))
                {
                    r.op_state.stop_source_.request_stop();
                }
                r.op_state.finish();
            };

            template <typename... Ts, std::size_t... Is>
            auto set_value_helper(hpx::util::index_pack<Is...>, Ts&&... ts)
            // MSVC sometimes invalidly rejects valid invocations
            // clang-format off
#if !defined(HPX_MSVC)
                -> decltype(
                    (std::declval<
                         typename OperationState::value_types_storage_type>()
                            .template get<OperationState::i_storage_offset +
                                Is>()
                            .emplace(HPX_FORWARD(Ts, ts)),
                        ...),
                    void())
#endif
            // clang-format on
            {
                // op_state.ts holds values from all predecessor senders. We
                // emplace the values using the offset calculated while
                // constructing the operation state.
                (op_state.ts
                        .template get<OperationState::i_storage_offset + Is>()
                        .emplace(HPX_FORWARD(Ts, ts)),
                    ...);
            }

            static constexpr std::size_t sender_pack_size =
                OperationState::sender_pack_size;
            using index_pack_type =
                hpx::util::make_index_pack_t<sender_pack_size>;

            // different versions of clang-format disagree
            // clang-format off
            template <typename... Ts>
            auto set_value(Ts&&... ts) noexcept -> decltype(
                set_value_helper(index_pack_type{}, HPX_FORWARD(Ts, ts)...))
            // clang-format on
            {
                if constexpr (sender_pack_size > 0)
                {
                    if (!op_state.set_stopped_error_called)
                    {
                        try
                        {
                            set_value_helper(
                                index_pack_type{}, HPX_FORWARD(Ts, ts)...);
                        }
                        catch (...)
                        {
                            if (!op_state.set_stopped_error_called.exchange(
                                    true))
                            {
                                // NOLINTNEXTLINE(bugprone-throw-keyword-missing)
                                op_state.error = std::current_exception();
                            }
                        }
                    }
                }

                op_state.finish();
            }

            friend auto tag_invoke(get_env_t, when_all_receiver const& r)
                -> make_env_t<get_stop_token_t,
                    hpx::experimental::in_place_stop_token,
                    env_of_t<typename OperationState::receiver_type>>
            {
                return make_env<get_stop_token_t>(
                    r.op_state.stop_source_.get_token(),
                    hpx::execution::experimental::get_env(r.op_state.receiver));
            }
        };

        // Due to what appears to be a bug in clang this is not a hidden friend
        // of when_all_receiver. The trailing decltype for SFINAE in the member
        // set_value would give an error about accessing an incomplete type, if
        // the member set_value were a hidden friend tag_invoke overload
        // instead.
        template <typename OperationState, typename... Ts>
        auto tag_invoke(set_value_t, when_all_receiver<OperationState>&& r,
            Ts&&... ts) noexcept
            -> decltype(r.set_value(HPX_FORWARD(Ts, ts)...), void())
        {
            r.set_value(HPX_FORWARD(Ts, ts)...);
        }

        template <typename... Senders>
        struct when_all_sender
        {
            using is_sender = void;
            using senders_type =
                hpx::util::member_pack_for<std::decay_t<Senders>...>;
            senders_type senders;

            template <typename... Senders_>
            explicit constexpr when_all_sender(Senders_&&... senders)
              : senders(
                    std::piecewise_construct, HPX_FORWARD(Senders_, senders)...)
            {
            }

            template <typename Env>
            struct generate_completion_signatures
            {
                template <template <typename...> typename Tuple,
                    template <typename...> typename Variant>
                using value_types = hpx::util::detail::concat_inner_packs_t<
                    hpx::util::detail::concat_t<
                        value_types_of_t<Senders, Env, Tuple, Variant>...>>;

                template <template <typename...> typename Variant>
                using error_types = hpx::util::detail::unique_concat_t<
                    error_types_of_t<Senders, Env, Variant>...,
                    Variant<std::exception_ptr>>;

                static constexpr bool sends_stopped = true;
            };

            template <typename Env>
            friend auto tag_invoke(get_completion_signatures_t,
                when_all_sender const&, Env) noexcept
                -> generate_completion_signatures<Env>;

            static constexpr std::size_t num_predecessors = sizeof...(Senders);
            static_assert(num_predecessors > 0,
                "when_all expects at least one predecessor sender");

            template <std::size_t I>
            static constexpr std::size_t sender_pack_size_at_index =
                single_variant_tuple_size_v<
                    value_types_of_t<hpx::util::at_index_t<I, Senders...>,
                        empty_env, meta::pack, meta::pack>>;

            template <typename Receiver, typename SendersPack,
                std::size_t I = num_predecessors - 1>
            struct operation_state;

#if defined(HPX_GCC_VERSION) && HPX_GCC_VERSION >= 110000
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Warray-bounds"
#endif
            template <typename Receiver, typename SendersPack>
            struct operation_state<Receiver, SendersPack, 0>
            {
                using receiver_type = std::decay_t<Receiver>;

                // The index of the sender that this operation state handles.
                static constexpr std::size_t i = 0;
                // The offset at which we start to emplace values sent by the
                // ith predecessor sender.
                static constexpr std::size_t i_storage_offset = 0;
#if !defined(HPX_CUDA_VERSION)
                // The number of values sent by the ith predecessor sender.
                static constexpr std::size_t sender_pack_size =
                    sender_pack_size_at_index<0>;
#else
                // nvcc does not like using the helper sender_pack_size_at_index
                // here and complains about incomplete types. Lifting the helper
                // explicitly in here works.
                static constexpr std::size_t sender_pack_size =
                    single_variant_tuple_size_v<
                        value_types_of_t<hpx::util::at_index_t<0, Senders...>,
                            empty_env, meta::pack, meta::pack>>;
#endif

                // Number of predecessor senders that have not yet called any of
                // the set signals.
                std::atomic<std::size_t> predecessors_remaining =
                    num_predecessors;

                // Values sent by all predecessor senders are stored here in the
                // base-case operation state. They are stored in a
                // member_pack<optional<T0>, ..., optional<Tn>>, where T0, ...,
                // Tn are the types of the values sent by all predecessor
                // senders.
                template <typename T>
                struct add_optional
                {
                    using type = hpx::optional<std::decay_t<T>>;
                };
                using value_types = typename generate_completion_signatures<
                    empty_env>::template value_types<meta::pack, meta::pack>;
                using value_types_storage_type =
                    hpx::util::detail::change_pack_t<hpx::util::member_pack_for,
                        hpx::util::detail::transform_t<
                            hpx::util::detail::concat_pack_of_packs_t<
                                value_types>,
                            add_optional>>;
                value_types_storage_type ts;

                using error_types = typename generate_completion_signatures<
                    empty_env>::template error_types<hpx::variant>;
                hpx::optional<error_types> error;
                std::atomic<bool> set_stopped_error_called{false};
                HPX_NO_UNIQUE_ADDRESS receiver_type receiver;

                hpx::experimental::in_place_stop_source stop_source_{};

                using stop_token_t = stop_token_of_t<env_of_t<receiver_type>&>;
                hpx::optional<typename stop_token_t::template callback_type<
                    on_stop_requested>>
                    on_stop_{};

                using operation_state_type =
                    std::decay_t<decltype(hpx::execution::experimental::connect(
                        std::declval<SendersPack>().template get<i>(),
                        when_all_receiver<operation_state>(
                            std::declval<std::decay_t<operation_state>&>())))>;
                operation_state_type op_state;

                template <typename Receiver_, typename Senders_>
                operation_state(Receiver_&& receiver, Senders_&& senders)
                  : receiver(HPX_FORWARD(Receiver_, receiver))
                  , op_state(hpx::execution::experimental::connect(
#if defined(HPX_CUDA_VERSION)
                        std::forward<Senders_>(senders).template get<i>(),
#else
                        HPX_FORWARD(Senders_, senders).template get<i>(),
#endif
                        when_all_receiver<operation_state>(*this)))
                {
                }

                operation_state(operation_state&&) = delete;
                operation_state& operator=(operation_state&&) = delete;
                operation_state(operation_state const&) = delete;
                operation_state& operator=(operation_state const&) = delete;

                void start() & noexcept
                {
                    hpx::execution::experimental::start(op_state);
                }

                template <std::size_t... Is, typename... Ts>
                void set_value_helper(
                    hpx::util::member_pack<hpx::util::index_pack<Is...>, Ts...>&
                        t) noexcept
                {
                    hpx::execution::experimental::set_value(
                        HPX_MOVE(receiver), HPX_MOVE(*t.template get<Is>())...);
                }

                void finish() noexcept
                {
                    if (--predecessors_remaining == 0)
                    {
                        // Stop callback is no longer needed. Destroy it.
                        on_stop_.reset();

                        if (!set_stopped_error_called)
                        {
                            set_value_helper(ts);
                        }
                        else if (error)
                        {
                            hpx::visit(
                                [this](auto&& error) {
                                    hpx::execution::experimental::set_error(
                                        HPX_MOVE(receiver),
                                        HPX_FORWARD(decltype(error), error));
                                },
                                HPX_MOVE(error.value()));
                        }
                        else
                        {
                            hpx::execution::experimental::set_stopped(
                                HPX_MOVE(receiver));
                        }
                    }
                }
            };

            template <typename Receiver, typename SendersPack, std::size_t I>
            struct operation_state
              : operation_state<Receiver, SendersPack, I - 1>
            {
                using base_type = operation_state<Receiver, SendersPack, I - 1>;
                using receiver_type = std::decay_t<Receiver>;

                // The index of the sender that this operation state handles.
                static constexpr std::size_t i = I;

#if !defined(HPX_CUDA_VERSION)
                // The number of values sent by the ith predecessor sender.
                static constexpr std::size_t sender_pack_size =
                    sender_pack_size_at_index<i>;
#else
                // nvcc does not like using the helper sender_pack_size_at_index
                // here and complains about incomplete types. Lifting the helper
                // explicitly in here works.
                static constexpr std::size_t sender_pack_size =
                    single_variant_tuple_size_v<
                        value_types_of_t<hpx::util::at_index_t<i, Senders...>,
                            empty_env, meta::pack, meta::pack>>;
#endif
                // The offset at which we start to emplace values sent by the
                // ith predecessor sender.
                static constexpr std::size_t i_storage_offset =
                    base_type::i_storage_offset + base_type::sender_pack_size;

                using operation_state_type =
                    std::decay_t<decltype(hpx::execution::experimental::connect(
                        std::declval<SendersPack>().template get<i>(),
                        when_all_receiver<operation_state>(
                            std::declval<std::decay_t<operation_state>&>())))>;
                operation_state_type op_state;

                template <typename Receiver_, typename SendersPack_>
                operation_state(Receiver_&& receiver, SendersPack_&& senders)
                  : base_type(HPX_FORWARD(Receiver_, receiver),
                        HPX_FORWARD(SendersPack, senders))
                  , op_state(hpx::execution::experimental::connect(
#if defined(HPX_CUDA_VERSION)
                        std::forward<SendersPack_>(senders).template get<i>(),
#else
                        HPX_FORWARD(SendersPack_, senders).template get<i>(),
#endif
                        when_all_receiver<operation_state>(*this)))
                {
                }

                operation_state(operation_state&&) = delete;
                operation_state& operator=(operation_state&&) = delete;
                operation_state(operation_state const&) = delete;
                operation_state& operator=(operation_state const&) = delete;

                void start() & noexcept
                {
                    base_type::start();
                    hpx::execution::experimental::start(op_state);
                }
            };
#if defined(HPX_GCC_VERSION) && HPX_GCC_VERSION >= 110000
#pragma GCC diagnostic pop
#endif

            template <typename Receiver, typename SendersPack>
            friend void tag_invoke(start_t,
                operation_state<Receiver, SendersPack, num_predecessors - 1>&
                    os) noexcept
            {
                // register stop callback
                os.on_stop_.emplace(
                    hpx::execution::experimental::get_stop_token(
                        hpx::execution::experimental::get_env(os.receiver)),
                    on_stop_requested{os.stop_source_});

                // If a stop has already been requested. Don't bother starting
                // the child operations.
                if (os.stop_source_.stop_requested())
                {
                    hpx::execution::experimental::set_stopped(
                        HPX_FORWARD(Receiver, os.receiver));
                }
                else
                {
                    os.start();
                }
            }

            template <typename Receiver>
            friend auto tag_invoke(
                connect_t, when_all_sender&& s, Receiver&& receiver)
            {
                return operation_state<Receiver, senders_type&&,
                    num_predecessors - 1>(
                    HPX_FORWARD(Receiver, receiver), HPX_MOVE(s.senders));
            }

            template <typename Receiver>
            friend auto tag_invoke(
                connect_t, when_all_sender& s, Receiver&& receiver)
            {
                return operation_state<Receiver, senders_type&,
                    num_predecessors - 1>(
                    HPX_FORWARD(Receiver, receiver), s.senders);
            }
        };
    }    // namespace detail

    // execution::when_all is used to join multiple sender chains and create a
    // sender whose execution is dependent on all of the input senders that only
    // send a single set of values. execution::when_all_with_variant is used to
    // join multiple sender chains and create a sender whose execution is
    // dependent on all of the input senders, each of which may have one or more
    // sets of sent values.
    //
    // when_all returns a sender that completes once all of the input senders
    // have completed. It is constrained to only accept senders that can
    // complete with a single set of values (_i.e._, it only calls one overload
    // of set_value on its receiver). The values sent by this sender are the
    // values sent by each of the input senders, in order of the arguments
    // passed to when_all. It completes inline on the execution context on which
    // the last input sender completes, unless stop is requested before when_all
    // is started, in which case it completes inline within the call to start.
    //
    // The returned sender has no completion schedulers.
    inline constexpr struct when_all_t final
      : hpx::functional::detail::tag_fallback<when_all_t>
    {
    private:
        // clang-format off
        template <typename... Senders,
            HPX_CONCEPT_REQUIRES_(
                hpx::util::all_of_v<is_sender<Senders>...>
            )>
        // clang-format on
        friend constexpr HPX_FORCEINLINE auto tag_fallback_invoke(
            when_all_t, Senders&&... senders)
        {
            return detail::when_all_sender<Senders...>{
                HPX_FORWARD(Senders, senders)...};
        }
    } when_all{};

    // TODO:
    // execution::when_all_with_variant is used to join multiple sender chains
    // and create a sender whose execution is dependent on all of the input
    // senders, each of which may have one or more sets of sent values.
    inline constexpr struct when_all_with_variant_t final
      : hpx::functional::tag<when_all_with_variant_t>
    {
    } when_all_with_variant{};

    // execution::transfer_when_all is used to join multiple sender chains
    // and create a sender whose execution is dependent on all of the input
    // senders that only send a single set of values each, while also making
    // sure that they complete on the specified scheduler.
    inline constexpr struct transfer_when_all_t final
      : hpx::functional::detail::tag_fallback<transfer_when_all_t>
    {
    private:
        // clang-format off
        template <typename Sched, typename... Senders,
            HPX_CONCEPT_REQUIRES_(
                is_scheduler_v<Sched> &&
                hpx::util::all_of_v<is_sender<Senders>...>
            )>
        // clang-format on
        friend constexpr HPX_FORCEINLINE auto tag_fallback_invoke(
            transfer_when_all_t, Sched&& sched, Senders&&... senders)
        {
            return hpx::execution::experimental::transfer(
                hpx::execution::experimental::when_all(
                    HPX_FORWARD(Senders, senders)...),
                HPX_FORWARD(Sched, sched));
        }
    } transfer_when_all{};

    // TODO:
    // execution::transfer_when_all_with_variant is used to join multiple
    // sender chains and create a sender whose execution is dependent on all
    // of the input senders, which may have one or more sets of sent values.
    inline constexpr struct transfer_when_all_with_variant_t final
      : hpx::functional::tag<transfer_when_all_with_variant_t>
    {
    } transfer_when_all_with_variant{};

    // the following enables directly using dataflow() with senders

    template <typename F, typename Sender, typename... Senders>
    constexpr HPX_FORCEINLINE auto tag_invoke(
        hpx::detail::dataflow_t, F&& f, Sender&& sender, Senders&&... senders)
        -> decltype(then(when_all(HPX_FORWARD(Sender, sender),
                             HPX_FORWARD(Senders, senders)...),
            HPX_FORWARD(F, f)))
    {
        return then(when_all(HPX_FORWARD(Sender, sender),
                        HPX_FORWARD(Senders, senders)...),
            HPX_FORWARD(F, f));
    }

    template <typename F, typename Sender, typename... Senders>
    constexpr HPX_FORCEINLINE auto tag_invoke(hpx::detail::dataflow_t,
        hpx::launch, F&& f, Sender&& sender, Senders&&... senders)
        -> decltype(then(when_all(HPX_FORWARD(Sender, sender),
                             HPX_FORWARD(Senders, senders)...),
            HPX_FORWARD(F, f)))
    {
        return then(when_all(HPX_FORWARD(Sender, sender),
                        HPX_FORWARD(Senders, senders)...),
            HPX_FORWARD(F, f));
    }
}    // namespace hpx::execution::experimental
