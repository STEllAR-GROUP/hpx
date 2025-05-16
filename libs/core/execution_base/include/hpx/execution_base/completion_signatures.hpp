//  Copyright (c) 2022-2023 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/concepts/concepts.hpp>
#include <hpx/datastructures/tuple.hpp>
#include <hpx/datastructures/variant.hpp>
#include <hpx/execution_base/get_env.hpp>
#include <hpx/execution_base/receiver.hpp>
#include <hpx/functional/detail/tag_fallback_invoke.hpp>
#include <hpx/functional/invoke_result.hpp>
#include <hpx/functional/tag_invoke.hpp>
#include <hpx/type_support/meta.hpp>
#include <hpx/type_support/pack.hpp>

#include <type_traits>
#include <utility>

#if defined(HPX_HAVE_CXX20_COROUTINES)
#include <hpx/assert.hpp>
#include <hpx/execution_base/operation_state.hpp>
#include <hpx/execution_base/traits/coroutine_traits.hpp>
#include <hpx/type_support/coroutines_support.hpp>

#include <exception>
#include <system_error>
#endif

#include <hpx/execution_base/stdexec_forward.hpp>

namespace hpx::execution::experimental {

    namespace hpxexec = hpx::execution::experimental;

    // empty_variant will be returned by execution::value_types_of_t and
    // execution::error_types_of_t if no signatures are provided.
    struct empty_variant
    {
        empty_variant() = delete;
    };

    namespace detail {

        // use this remove_cv_ref instead of std::decay to avoid
        // decaying function types, e.g. set_value_t() -> set_value_t(*)()
        template <typename T>
        struct remove_cv_ref
        {
            using type = std::remove_cv_t<std::remove_reference_t<T>>;
        };

        template <typename T>
        using remove_cv_ref_t = meta::type<remove_cv_ref<T>>;

        // clang-format off

        // If sizeof...(Ts) is greater than zero, variant-or-empty<Ts...> names
        // the type variant<Us...> where Us... is the pack decay_t<Ts>... with
        // duplicate types removed.
        template <template <typename...> typename Variant>
        struct decay_variant_or_empty
        {
            template <typename... Ts>
            using apply = meta::invoke<
                meta::if_<meta::bool_<sizeof...(Ts) != 0>,
                    meta::transform<
                        meta::func1<remove_cv_ref_t>,
                        meta::unique<meta::func<Variant>>>,
                    meta::constant<empty_variant>>,
                Ts...>;
        };

        template <template <typename...> typename Variant>
        struct decay_variant
        {
            template <typename... Ts>
            using apply = meta::invoke<
                meta::transform<
                    meta::func1<remove_cv_ref_t>,
                    meta::unique<meta::func<Variant>>>,
                Ts...>;
        };

        template <template <typename...> typename Variant>
        struct unique_variant
        {
            template <typename... Ts>
            using apply =
                meta::invoke<meta::unique<meta::func<Variant>>, Ts...>;
        };
        // clang-format on

        template <typename... Ts>
        using decayed_variant =
            meta::invoke<decay_variant_or_empty<hpx::variant>, Ts...>;

        template <template <typename...> typename Tuple>
        struct decay_tuple
        {
            template <typename... Ts>
            using apply = Tuple<remove_cv_ref_t<Ts>...>;
        };

        template <typename... Ts>
        using decayed_tuple = meta::invoke<decay_tuple<hpx::tuple>, Ts...>;

    }    // namespace detail

    namespace detail {

        template <typename Sender, typename Enable = void>
        struct has_completion_signatures : std::false_type
        {
        };

        template <typename Sender>
        struct has_completion_signatures<Sender,
            std::void_t<
                typename remove_cv_ref_t<Sender>::completion_signatures>>
          : std::true_type
        {
        };
    }    // namespace detail

    // A sender is a type that is describing an asynchronous operation. The
    // operation itself might not have started yet. In order to get the result
    // of this asynchronous operation, a sender needs to be connected to a
    // receiver with the corresponding value, error and stopped channels:
    //
    //     * `hpx::execution::experimental::connect`
    //
    // A sender describes a potentially asynchronous operation. A sender's
    // responsibility is to fulfill the receiver contract of a connected
    // receiver by delivering one of the receiver completion-signals.
    //
    // A sender's destructor shall not block pending completion of submitted
    // operations.
    template <typename Sender, typename... Env>
    struct is_sender_in : std::bool_constant<hpxexec::sender_in<Sender, Env...>>
    {
    };

    template <typename Sender, typename... Env>
    inline constexpr bool is_sender_in_v = is_sender_in<Sender, Env...>::value;

    template <typename Sender>
    struct is_sender : std::bool_constant<hpxexec::sender<Sender>>
    {
    };

    template <typename Sender>
    inline constexpr bool is_sender_v = is_sender<Sender>::value;

    // \see is_sender
    template <typename Sender, typename Receiver>
    struct is_sender_to
      : std::bool_constant<hpxexec::sender_to<Sender, Receiver>>
    {
    };

    template <typename Sender, typename Receiver>
    inline constexpr bool is_sender_to_v =
        is_sender_to<Sender, Receiver>::value;

    // The sender_of concept defines the requirements for a sender type that on
    // successful completion sends the specified set of value types.
    template <typename Sender, typename Signal,
        typename Env = hpxexec::empty_env>
    struct is_sender_of
      : std::bool_constant<hpxexec::sender_of<Sender, Signal, Env>>
    {
    };

    template <typename Sender, typename Signal,
        typename Env = hpxexec::empty_env>
    inline constexpr bool is_sender_of_v =
        is_sender_of<Sender, Signal, Env>::value;

    // As outlined in the original implementation:
    //
    // Alias template single-sender-value-type is defined as follows:
    //
    // 1. If value_types_of_t<S, E, Tuple, Variant> would have the form
    // Variant<Tuple<T>>, then single-sender-value-type<S, E> is an alias for
    // type T.
    // 2. Otherwise, if value_types_of_t<S, E, Tuple, Variant> would
    // have the form Variant<Tuple<>> or Variant<>, then
    // single-sender-value-type<S, E> is an alias for type void.
    // 3. Otherwise, single-sender-value-type<S, E> is ill-formed.
    //
    template <typename Sender, typename Env = hpxexec::empty_env>
    using single_sender_value_t =
        hpxexec::stdexec_internal::__single_sender_value_t<Sender, Env>;

    template <typename A, typename B>
    inline constexpr bool is_derived_from_v = std::derived_from<A, B>;

    // Does not exist in P2300
    //    template<typename Promise>
    //    using with_awaitable_senders =
    //    hpx::execution::experimental::with_awaitable_senders<Promise>;

    template <typename ReceiverID>
    using operation =
        hpxexec::stdexec_internal::__connect_awaitable_::__operation<
            ReceiverID>;

    template <typename ReceiverID>
    using promise =
        hpxexec::stdexec_internal::__connect_awaitable_::__promise<ReceiverID>;

    template <typename Rec>
    using promise_t =
        hpxexec::stdexec_internal::__connect_awaitable_::__promise_t<Rec>;

    template <typename Rec>
    using operation_t =
        hpxexec::stdexec_internal::__connect_awaitable_::__operation_t<Rec>;

    using connect_awaitable_t =
        hpxexec::stdexec_internal::__connect_awaitable_t;
    inline constexpr connect_awaitable_t connect_awaitable{};

    template <typename Sender, typename Receiver>
    struct has_nothrow_connect
      : std::integral_constant<bool,
            noexcept(hpxexec::connect(
                std::declval<Sender>(), std::declval<Receiver>()))>
    {
    };

}    // namespace hpx::execution::experimental
