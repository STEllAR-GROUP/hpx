//  Copyright (c) 2020 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/local/config.hpp>
#include <hpx/concepts/concepts.hpp>
#include <hpx/datastructures/optional.hpp>
#include <hpx/datastructures/variant.hpp>
#include <hpx/execution/algorithms/detail/single_result.hpp>
#include <hpx/execution_base/operation_state.hpp>
#include <hpx/execution_base/receiver.hpp>
#include <hpx/execution_base/sender.hpp>
#include <hpx/functional/invoke_fused.hpp>
#include <hpx/functional/tag_fallback_dispatch.hpp>
#include <hpx/type_support/pack.hpp>

#include <atomic>
#include <cstddef>
#include <exception>
#include <functional>
#include <memory>
#include <type_traits>
#include <utility>

namespace hpx { namespace execution { namespace experimental {
    namespace detail {
        template <std::size_t I, typename OperationState>
        struct when_all_receiver
        {
            std::decay_t<OperationState>& op_state;

            when_all_receiver(std::decay_t<OperationState>& op_state)
              : op_state(op_state)
            {
            }

            template <typename Error>
            friend void tag_dispatch(
                set_error_t, when_all_receiver&& r, Error&& error) noexcept
            {
                if (!r.op_state.set_done_error_called.exchange(true))
                {
                    try
                    {
                        r.op_state.error = std::forward<Error>(error);
                    }
                    catch (...)
                    {
                        // NOLINTNEXTLINE(bugprone-throw-keyword-missing)
                        r.op_state.error = std::current_exception();
                    }
                }

                r.op_state.finish();
            }

            friend void tag_dispatch(set_done_t, when_all_receiver&& r) noexcept
            {
                r.op_state.set_done_error_called = true;
                r.op_state.finish();
            };

            template <typename T>
            friend void tag_dispatch(
                set_value_t, when_all_receiver&& r, T&& t) noexcept
            {
                if (!r.op_state.set_done_error_called)
                {
                    try
                    {
                        r.op_state.ts.template get<I>().emplace(
                            std::forward<T>(t));
                    }
                    catch (...)
                    {
                        if (!r.op_state.set_done_error_called.exchange(true))
                        {
                            // NOLINTNEXTLINE(bugprone-throw-keyword-missing)
                            r.op_state.error = std::current_exception();
                        }
                    }
                }

                r.op_state.finish();
            }
        };

        template <typename... Senders>
        struct when_all_sender
        {
            using senders_type =
                hpx::util::member_pack_for<std::decay_t<Senders>...>;
            senders_type senders;

            template <typename... Senders_>
            explicit constexpr when_all_sender(Senders_&&... senders)
              : senders(std::piecewise_construct,
                    std::forward<Senders_>(senders)...)
            {
            }

            template <typename Sender>
            struct value_types_helper
            {
                using value_types =
                    typename hpx::execution::experimental::sender_traits<
                        Sender>::template value_types<hpx::util::pack,
                        hpx::util::pack>;
                using type = detail::single_result_non_void_t<value_types>;
            };

            template <typename Sender>
            using value_types_helper_t =
                typename value_types_helper<Sender>::type;

            template <template <typename...> class Tuple,
                template <typename...> class Variant>
            using value_types =
                Variant<Tuple<value_types_helper_t<Senders>...>>;

            template <template <typename...> class Variant>
            using error_types = hpx::util::detail::unique_concat_t<
                typename hpx::execution::experimental::sender_traits<
                    Senders>::template error_types<Variant>...,
                Variant<std::exception_ptr>>;

            static constexpr bool sends_done = false;

            static constexpr std::size_t num_predecessors = sizeof...(Senders);
            static_assert(num_predecessors > 0,
                "when_all expects at least one predecessor sender");

            template <typename Receiver, typename SendersPack, std::size_t I>
            struct operation_state;

            template <typename Receiver, typename SendersPack>
            struct operation_state<Receiver, SendersPack, 0>
            {
                static constexpr std::size_t I = 0;
                std::atomic<std::size_t> predecessors_remaining =
                    num_predecessors;
                hpx::util::member_pack_for<hpx::optional<
                    std::decay_t<value_types_helper_t<Senders>>>...>
                    ts;
                hpx::optional<error_types<hpx::variant>> error;
                std::atomic<bool> set_done_error_called{false};
                HPX_NO_UNIQUE_ADDRESS std::decay_t<Receiver> receiver;

                using operation_state_type =
                    std::decay_t<decltype(hpx::execution::experimental::connect(
                        std::declval<SendersPack>().template get<I>(),
                        when_all_receiver<I, operation_state>(
                            std::declval<std::decay_t<operation_state>&>())))>;
                operation_state_type op_state;

                template <typename Receiver_, typename Senders_>
                operation_state(Receiver_&& receiver, Senders_&& senders)
                  : receiver(std::forward<Receiver_>(receiver))
                  , op_state(hpx::execution::experimental::connect(
                        std::forward<Senders_>(senders).template get<I>(),
                        when_all_receiver<I, operation_state>(*this)))
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
                        ts)
                {
                    hpx::execution::experimental::set_value(std::move(receiver),
                        std::move(*(ts.template get<Is>()))...);
                }

                void finish() noexcept
                {
                    if (--predecessors_remaining == 0)
                    {
                        if (!set_done_error_called)
                        {
                            set_value_helper(ts);
                        }
                        else if (error)
                        {
                            hpx::visit(
                                [this](auto&& error) {
                                    hpx::execution::experimental::set_error(
                                        std::move(receiver),
                                        std::forward<decltype(error)>(error));
                                },
                                std::move(error.value()));
                        }
                        else
                        {
                            hpx::execution::experimental::set_done(
                                std::move(receiver));
                        }
                    }
                }
            };

            template <typename Receiver, typename SendersPack, std::size_t I>
            struct operation_state
              : operation_state<Receiver, SendersPack, I - 1>
            {
                using base_type = operation_state<Receiver, SendersPack, I - 1>;

                using operation_state_type =
                    std::decay_t<decltype(hpx::execution::experimental::connect(
                        std::forward<SendersPack>(senders).template get<I>(),
                        when_all_receiver<I, operation_state>(
                            std::declval<std::decay_t<operation_state>&>())))>;
                operation_state_type op_state;

                template <typename Receiver_, typename SendersPack_>
                operation_state(Receiver_&& receiver, SendersPack_&& senders)
                  : base_type(std::forward<Receiver_>(receiver),
                        std::forward<SendersPack>(senders))
                  , op_state(hpx::execution::experimental::connect(
                        std::forward<SendersPack_>(senders).template get<I>(),
                        when_all_receiver<I, operation_state>(*this)))
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

            template <typename Receiver, typename SendersPack>
            friend void tag_dispatch(start_t,
                operation_state<Receiver, SendersPack, num_predecessors - 1>&
                    os) noexcept
            {
                os.start();
            }

            template <typename Receiver>
            friend auto tag_dispatch(
                connect_t, when_all_sender&& s, Receiver&& receiver)
            {
                return operation_state<Receiver, senders_type&&,
                    num_predecessors - 1>(
                    std::forward<Receiver>(receiver), std::move(s.senders));
            }

            template <typename Receiver>
            friend auto tag_dispatch(
                connect_t, when_all_sender& s, Receiver&& receiver)
            {
                return operation_state<Receiver, senders_type&,
                    num_predecessors - 1>(receiver, s.senders);
            }
        };
    }    // namespace detail

    HPX_INLINE_CONSTEXPR_VARIABLE struct when_all_t final
      : hpx::functional::tag_fallback<when_all_t>
    {
    private:
        // clang-format off
        template <typename... Senders,
            HPX_CONCEPT_REQUIRES_(
                hpx::util::all_of_v<is_sender<Senders>...>
            )>
        // clang-format on
        friend constexpr HPX_FORCEINLINE auto tag_fallback_dispatch(
            when_all_t, Senders&&... senders)
        {
            return detail::when_all_sender<Senders...>{
                std::forward<Senders>(senders)...};
        }
    } when_all{};
}}}    // namespace hpx::execution::experimental
