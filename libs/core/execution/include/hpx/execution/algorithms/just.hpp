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

        template <typename CPO, typename... Ts>
        using just_completion_signatures =
            completion_signatures<CPO(Ts...), set_error_t(std::exception_ptr)>;

        template <typename CPO, typename... Ts>
        struct just_sender_base;

        template <typename ReceiverId, typename CPO, typename Is,
            typename... Ts>
        struct just_sender_operation_state;

        template <typename ReceiverId, typename CPO, std::size_t... Is,
            typename... Ts>
        struct just_sender_operation_state<ReceiverId, CPO,
            hpx::util::index_pack<Is...>, Ts...>
        {
            using Receiver = hpx::meta::type<ReceiverId>;

            struct type
            {
                using id = just_sender_operation_state;
                using data_type = std::tuple<Ts...>;

                // type() = default;

                HPX_NO_UNIQUE_ADDRESS data_type ts;
                HPX_NO_UNIQUE_ADDRESS Receiver receiver;

                friend void tag_invoke(start_t, type& os) noexcept
                {
                    hpx::detail::try_catch_exception_ptr(
                        [&]() {
                            CPO{}(HPX_FORWARD(Receiver, os.receiver),
                                std::get<Is>(HPX_FORWARD(Ts, os.ts))...);
                        },
                        [&](std::exception_ptr ep) {
                            hpx::execution::experimental::set_error(
                                HPX_MOVE(os.receiver), HPX_MOVE(ep));
                        });
                }

                // private:
                //     type(type&&) = delete;
            };
        };

        template <typename CPO, typename... Ts>
        struct just_sender_base
        {
            using Is = hpx::util::make_index_pack_t<sizeof...(Ts)>;

            template <typename Receiver>
            using just_operation_t =
                hpx::meta::type<just_sender_operation_state<
                    hpx::meta::get_id_t<Receiver>, CPO, Is, Ts...>>;

            struct type
            {
                using is_sender = void;
                using id = just_sender_base;
                using completion_signatures =
                    just_completion_signatures<CPO, Ts...>;

                HPX_NO_UNIQUE_ADDRESS std::tuple<Ts...> ts;

                // constexpr type() = default;

                // type(type&&) = default;
                // type(type const&) = default;
                // type& operator=(type&&) = default;
                // type& operator=(type const&) = default;

                template <typename Receiver>
                friend auto
                tag_invoke(connect_t, type&& s, Receiver receiver) noexcept(
                    util::all_of_v<std::is_nothrow_move_constructible<Ts>...>)
                {
                    return just_operation_t<Receiver>{
                        HPX_MOVE(s.ts), HPX_FORWARD(Receiver, receiver)};
                }

                template <typename Receiver>
                friend auto tag_invoke(
                    connect_t, const type& s, Receiver receiver) noexcept(util::
                        all_of_v<std::is_nothrow_copy_constructible<Ts>...>)
                {
                    return just_operation_t<Receiver>{
                        s.ts, HPX_FORWARD(Receiver, receiver)};
                }
            };
        };

        template <typename... Values>
        struct just_sender
        {
            using base =
                hpx::meta::type<just_sender_base<set_value_t, Values...>>;

            struct type : base
            {
                using id = just_sender;
            };
        };

        template <typename Error>
        struct just_error_sender
        {
            using base = hpx::meta::type<just_sender_base<set_error_t, Error>>;

            struct type : base
            {
                using id = just_error_sender;
            };
        };

        struct just_stopped_sender
          : hpx::meta::type<just_sender_base<set_stopped_t>>
        {
            using id = just_stopped_sender;
            using type = just_stopped_sender;
        };
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
            -> hpx::meta::type<detail::just_sender<std::decay_t<Ts>...>>
        {
            return {{{HPX_FORWARD(Ts, ts)...}}};
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
            -> hpx::meta::type<detail::just_error_sender<std::decay_t<Error>>>
        {
            return {{{HPX_FORWARD(Error, error)}}};
        }
    } just_error{};

    // Returns a sender with no completion schedulers, which completes
    // immediately by calling the receiver's set_stopped.
    inline constexpr struct just_stopped_t final
    {
        template <typename... Ts>
        constexpr HPX_FORCEINLINE auto operator()() const
            -> hpx::meta::type<detail::just_stopped_sender>
        {
            return {{}};
        }
    } just_stopped{};
}    // namespace hpx::execution::experimental
