//  Copyright (c) 2020 ETH Zurich
//  Copyright (c) 2022 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/datastructures/member_pack.hpp>
#include <hpx/errors/try_catch_exception_ptr.hpp>
#include <hpx/execution_base/completion_signatures.hpp>
#include <hpx/execution_base/receiver.hpp>
#include <hpx/execution_base/sender.hpp>
#include <hpx/type_support/pack.hpp>

#include <cstddef>
#include <exception>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace hpx::execution::experimental {

    namespace detail {

        template <typename CPO, typename Is, typename... Ts>
        struct just_sender;

        template <typename CPO, std::size_t... Is, typename... Ts>
        struct just_sender<CPO, hpx::util::index_pack<Is...>, Ts...>
        {
            using is_sender = void;

            HPX_NO_UNIQUE_ADDRESS util::member_pack_for<std::decay_t<Ts>...> ts;

            constexpr just_sender() = default;

            template <typename T,
                typename = std::enable_if_t<
                    !std::is_same_v<std::decay_t<T>, just_sender>>>
            explicit constexpr just_sender(T&& t)
              : ts(std::piecewise_construct, HPX_FORWARD(T, t))
            {
            }

            template <typename T0, typename T1, typename... Ts_>
            explicit constexpr just_sender(T0&& t0, T1&& t1, Ts_&&... ts)
              : ts(std::piecewise_construct, HPX_FORWARD(T0, t0),
                    HPX_FORWARD(T1, t1), HPX_FORWARD(Ts_, ts)...)
            {
            }

            just_sender(just_sender&&) = default;
            just_sender(just_sender const&) = default;
            just_sender& operator=(just_sender&&) = default;
            just_sender& operator=(just_sender const&) = default;

#if defined(HPX_GCC_VERSION) && HPX_GCC_VERSION >= 110000
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Warray-bounds"
#endif
            template <typename Receiver>
            struct operation_state
            {
                using data_type =
                    hpx::util::member_pack_for<std::decay_t<Ts>...>;

                HPX_NO_UNIQUE_ADDRESS std::decay_t<Receiver> receiver;
                HPX_NO_UNIQUE_ADDRESS
                data_type ts;

                template <typename Receiver_>
                operation_state(Receiver_&& receiver, data_type ts)
                  : receiver(HPX_FORWARD(Receiver_, receiver))
                  , ts(HPX_MOVE(ts))
                {
                }

                operation_state(operation_state&&) = delete;
                operation_state& operator=(operation_state&&) = delete;
                operation_state(operation_state const&) = delete;
                operation_state& operator=(operation_state const&) = delete;

                friend void tag_invoke(start_t, operation_state& os) noexcept
                {
                    hpx::detail::try_catch_exception_ptr(
                        [&]() {
                            CPO{}(HPX_MOVE(os.receiver),
                                HPX_MOVE(os.ts).template get<Is>()...);
                        },
                        [&](std::exception_ptr ep) {
                            hpx::execution::experimental::set_error(
                                HPX_MOVE(os.receiver), HPX_MOVE(ep));
                        });
                }
            };
#if defined(HPX_GCC_VERSION) && HPX_GCC_VERSION >= 110000
#pragma GCC diagnostic pop
#endif

            template <typename Receiver>
            friend auto tag_invoke(
                connect_t, just_sender&& s, Receiver&& receiver) noexcept(util::
                    all_of_v<std::is_nothrow_move_constructible<Ts>...>)
            {
                return operation_state<Receiver>{
                    HPX_FORWARD(Receiver, receiver), HPX_MOVE(s.ts)};
            }

            template <typename Receiver>
            friend auto
            tag_invoke(connect_t, just_sender& s, Receiver&& receiver) noexcept(
                util::all_of_v<std::is_nothrow_copy_constructible<Ts>...>)
            {
                return operation_state<Receiver>{
                    HPX_FORWARD(Receiver, receiver), s.ts};
            }
        };

        template <typename Pack, typename... Ts, typename Env>
        auto tag_invoke(get_completion_signatures_t,
            just_sender<set_value_t, Pack, Ts...> const&, Env) noexcept
            -> hpx::execution::experimental::completion_signatures<
                set_value_t(Ts...), set_error_t(std::exception_ptr)>;

        // different versions of clang-format disagree
        // clang-format off
        template <typename Pack, typename Error, typename Env>
        auto tag_invoke(get_completion_signatures_t,
            just_sender<set_error_t, Pack, Error> const&, Env) noexcept
            -> hpx::execution::experimental::completion_signatures<
                set_error_t(std::exception_ptr), set_error_t(Error)>;
        // clang-format on

        template <typename Pack, typename... Ts, typename Env>
        auto tag_invoke(get_completion_signatures_t,
            just_sender<set_stopped_t, Pack, Ts...> const&, Env) noexcept
            -> hpx::execution::experimental::completion_signatures<
                set_stopped_t(), set_error_t(std::exception_ptr)>;
    }    // namespace detail

    // Returns a sender with no completion schedulers, which sends the provided
    // values. The input values are decay-copied into the returned sender. When
    // the returned sender is connected to a receiver, the values are moved into
    // the operation state if the sender is an rvalue; otherwise, they are
    // copied. Then xvalues referencing the values in the operation state are
    // passed to the receiver's set_value.
    inline constexpr struct just_t final
    {
        template <typename... Ts>
        constexpr HPX_FORCEINLINE auto operator()(Ts&&... ts) const
        {
            return detail::just_sender<
                hpx::execution::experimental::set_value_t,
                hpx::util::make_index_pack_t<sizeof...(Ts)>, Ts...>{
                HPX_FORWARD(Ts, ts)...};
        }
    } just{};

    // Returns a sender with no completion schedulers, which completes with the
    // specified error. If the provided error is an lvalue reference, a copy is
    // made inside the returned sender and a non-const lvalue reference to the
    // copy is sent to the receiver's set_error. If the provided value is an
    // rvalue reference, it is moved into the returned sender and an rvalue
    // reference to it is sent to the receiver's set_error.
    inline constexpr struct just_error_t final
    {
        template <typename Error>
        constexpr HPX_FORCEINLINE auto operator()(Error&& error) const
        {
            return detail::just_sender<
                hpx::execution::experimental::set_error_t,
                hpx::util::make_index_pack_t<std::size_t(1)>, Error>{
                HPX_FORWARD(Error, error)};
        }
    } just_error{};

    // Returns a sender with no completion schedulers, which completes
    // immediately by calling the receiver's set_stopped.
    inline constexpr struct just_stopped_t final
    {
        template <typename... Ts>
        constexpr HPX_FORCEINLINE auto operator()(Ts&&... ts) const
        {
            return detail::just_sender<
                hpx::execution::experimental::set_stopped_t,
                hpx::util::make_index_pack_t<sizeof...(Ts)>, Ts...>{
                HPX_FORWARD(Ts, ts)...};
        }
    } just_stopped{};
}    // namespace hpx::execution::experimental
