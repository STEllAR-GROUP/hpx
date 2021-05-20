//  Copyright (c) 2020 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#if defined(HPX_HAVE_CXX17_STD_VARIANT)
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
#include <optional>
#include <type_traits>
#include <utility>
#include <variant>

namespace hpx { namespace execution { namespace experimental {
    namespace detail {
        template <std::size_t I, typename OS>
        struct when_all_receiver
        {
            std::decay_t<OS>& os;

            when_all_receiver(std::decay_t<OS>& os)
              : os(os)
            {
            }

            template <typename E>
                void set_error(E&& e) && noexcept
            {
                if (!os.set_done_error_called.exchange(true))
                {
                    try
                    {
                        os.e = std::forward<E>(e);
                    }
                    catch (...)
                    {
                        os.e = std::current_exception();
                    }
                }

                os.finish();
            }

            void set_done() && noexcept
            {
                os.set_done_error_called = true;
                os.finish();
            };

            template <typename T>
                void set_value(T&& t) && noexcept
            {
                if (!os.set_done_error_called)
                {
                    try
                    {
                        os.ts.template get<I>().emplace(std::forward<T>(t));
                    }
                    catch (...)
                    {
                        if (!os.set_done_error_called.exchange(true))
                        {
                            os.e = std::current_exception();
                        }
                    }
                }

                os.finish();
            }
        };

        template <typename... Ss>
        struct when_all_sender
        {
            hpx::util::member_pack_for<std::decay_t<Ss>...> senders;

            template <typename... Ss_>
            explicit constexpr when_all_sender(Ss_&&... ss)
              : senders(std::piecewise_construct, std::forward<Ss_>(ss)...)
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
            using value_types = Variant<Tuple<value_types_helper_t<Ss>...>>;

            template <template <typename...> class Variant>
            using error_types = hpx::util::detail::unique_concat_t<
                typename hpx::execution::experimental::sender_traits<
                    Ss>::template error_types<Variant>...,
                Variant<std::exception_ptr>>;

            static constexpr bool sends_done = false;

            static constexpr std::size_t num_predecessors = sizeof...(Ss);
            static_assert(num_predecessors > 0,
                "when_all expects at least one predecessor sender");

            template <typename R, std::size_t I>
            struct operation_state;

            template <typename R>
            struct operation_state<R, 0>
            {
                static constexpr std::size_t I = 0;
                std::atomic<std::size_t> predecessors_remaining =
                    num_predecessors;
                hpx::util::member_pack_for<
                    std::optional<std::decay_t<value_types_helper_t<Ss>>>...>
                    ts;
                std::optional<error_types<std::variant>> e;
                std::atomic<bool> set_done_error_called{false};
                std::decay_t<R> r;

                using operation_state_type =
                    std::decay_t<decltype(hpx::execution::experimental::connect(
                        std::move(senders.template get<I>()),
                        when_all_receiver<I, operation_state>(
                            std::declval<std::decay_t<operation_state>&>())))>;
                operation_state_type os;

                template <typename R_>
                operation_state(R_&& r,
                    hpx::util::member_pack_for<std::decay_t<Ss>...>& senders)
                  : r(std::forward<R_>(r))
                  , os(hpx::execution::experimental::connect(
                        std::move(senders.template get<I>()),
                        when_all_receiver<I, operation_state>(*this)))
                {
                }

                operation_state(operation_state&&) = delete;
                operation_state& operator=(operation_state&&) = delete;
                operation_state(operation_state const&) = delete;
                operation_state& operator=(operation_state const&) = delete;

                void start() & noexcept
                {
                    hpx::execution::experimental::start(os);
                }

                template <std::size_t... Is, typename... Ts>
                void set_value_helper(
                    hpx::util::member_pack<hpx::util::index_pack<Is...>, Ts...>&
                        ts)
                {
                    hpx::execution::experimental::set_value(
                        std::move(r), std::move(*(ts.template get<Is>()))...);
                }

                void finish() noexcept
                {
                    if (--predecessors_remaining == 0)
                    {
                        if (!set_done_error_called)
                        {
                            set_value_helper(ts);
                        }
                        else if (e)
                        {
                            std::visit(
                                [this](auto&& e) {
                                    hpx::execution::experimental::set_error(
                                        std::move(r),
                                        std::forward<decltype(e)>(e));
                                },
                                std::move(e.value()));
                        }
                        else
                        {
                            hpx::execution::experimental::set_done(
                                std::move(r));
                        }
                    }
                }
            };

            template <typename R, std::size_t I>
            struct operation_state : operation_state<R, I - 1>
            {
                using base_type = operation_state<R, I - 1>;

                using operation_state_type =
                    std::decay_t<decltype(hpx::execution::experimental::connect(
                        std::move(senders.template get<I>()),
                        when_all_receiver<I, operation_state>(
                            std::declval<std::decay_t<operation_state>&>())))>;
                operation_state_type os;

                template <typename R_>
                operation_state(R_&& r,
                    hpx::util::member_pack_for<std::decay_t<Ss>...>& senders)
                  : base_type(std::forward<R_>(r), senders)
                  , os(hpx::execution::experimental::connect(
                        std::move(senders.template get<I>()),
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
                    hpx::execution::experimental::start(os);
                }
            };

            template <typename R>
            auto connect(R&& r) &&
            {
                return operation_state<R, num_predecessors - 1>(
                    std::forward<R>(r), senders);
            }

            template <typename R>
            auto connect(R&& r) &
            {
                return operation_state<R, num_predecessors - 1>(r, senders);
            }
        };
    }    // namespace detail

    HPX_INLINE_CONSTEXPR_VARIABLE struct when_all_t final
      : hpx::functional::tag_fallback<when_all_t>
    {
    private:
        template <typename... Ss>
        friend constexpr HPX_FORCEINLINE auto tag_fallback_dispatch(
            when_all_t, Ss&&... ss)
        {
            return detail::when_all_sender<Ss...>{std::forward<Ss>(ss)...};
        }
    } when_all{};
}}}    // namespace hpx::execution::experimental
#endif
