//  Copyright (c) 2022-2023 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/type_support/detected.hpp>
#include <hpx/type_support/identity.hpp>
#include <hpx/type_support/pack.hpp>

#include <cstddef>
#include <type_traits>

namespace hpx::meta {

    ///////////////////////////////////////////////////////////////////////////
    // make an alias template that extracts the embedded ::type of T
    template <typename T>
    using type = typename T::type;

    // hide a template type parameter from ADL
    namespace detail {

        template <typename T>
        struct hidden
        {
            using type = struct _
            {
                using type = T;
            };
        };
    }    // namespace detail

    template <typename T>
    using hidden = meta::type<detail::hidden<T>>;

    ///////////////////////////////////////////////////////////////////////////
    template <typename T>
    using identity = T;

    template <typename T>
    inline constexpr bool value = T::value;

    template <typename T, typename U>
    inline constexpr bool value<std::is_same<T, U>> = false;

    template <typename T>
    inline constexpr bool value<std::is_same<T, T>> = true;

    ///////////////////////////////////////////////////////////////////////////
    template <template <class...> typename F>
    struct func
    {
        template <typename... Ts>
        using apply = F<Ts...>;
    };

    template <template <class> typename F>
    struct func1
    {
        template <typename T1>
        using apply = F<T1>;
    };

    template <template <class, class> typename F>
    struct func2
    {
        template <typename T1, typename T2>
        using apply = F<T1, T2>;
    };

    template <template <class, class, class> typename F>
    struct func3
    {
        template <typename T1, typename T2, typename T3>
        using apply = F<T1, T2, T3>;
    };

    template <typename R>
    struct compose_func
    {
        template <typename... Ts>
        using apply = R(Ts...);
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename F, typename... Ts>
    using invoke = typename F::template apply<Ts...>;

    template <typename F, typename T1>
    using invoke1 = typename F::template apply<T1>;

    template <typename F, typename T1, typename T2>
    using invoke2 = typename F::template apply<T1, T2>;

    template <typename F, typename T1, typename T2, typename T3>
    using invoke3 = typename F::template apply<T1, T2, T3>;

    ///////////////////////////////////////////////////////////////////////////
    template <template <class...> typename T, typename... Ts>
    using is_valid = util::is_detected<T, Ts...>;

    template <template <class> typename T, typename T1>
    using is_valid1 = util::is_detected<T, T1>;

    template <template <class, class> typename T, typename T1, typename T2>
    using is_valid2 = util::is_detected<T, T1, T2>;

    template <template <class, class, class> typename T, typename T1,
        typename T2, typename T3>
    using is_valid3 = util::is_detected<T, T1, T2, T3>;

    ///////////////////////////////////////////////////////////////////////////
    template <typename F, typename... Ts>
    using is_invocable = is_valid<F::template apply, Ts...>;

    template <typename F, typename T1>
    using is_invocable1 = is_valid1<F::template apply, T1>;

    template <typename F, typename T1, typename T2>
    using is_invocable2 = is_valid2<F::template apply, T1, T2>;

    template <typename F, typename T1, typename T2, typename T3>
    using is_invocable3 = is_valid3<F::template apply, T1, T2, T3>;

    ///////////////////////////////////////////////////////////////////////////
    // intentionally left unimplemented
    template <typename... Ts>
    struct pack;

    template <template <class...> typename F, typename... Front>
    struct bind_front_func
    {
        template <typename... Ts>
        using apply = F<Front..., Ts...>;
    };

    template <template <class...> typename F, typename... Front>
    struct bind_front1_func
    {
        template <typename A>
        using apply = F<Front..., A>;
    };

    template <template <class...> typename F, typename... Front>
    struct bind_front2_func
    {
        template <typename A, typename B>
        using apply = F<Front..., A, B>;
    };

    template <template <class...> typename F, typename... Front>
    struct bind_front3_func
    {
        template <typename A, typename B, typename C>
        using apply = F<Front..., A, B, C>;
    };

    template <typename F, typename... Front>
    using bind_front = bind_front_func<F::template apply, Front...>;

    template <typename F, typename... Back>
    using bind_front1 = bind_front1_func<F::template apply, Back...>;

    template <typename F, typename... Back>
    using bind_front2 = bind_front2_func<F::template apply, Back...>;

    template <typename F, typename... Back>
    using bind_front3 = bind_front3_func<F::template apply, Back...>;

    template <template <class...> typename F, typename... Back>
    struct bind_back_func
    {
        template <typename... Ts>
        using apply = F<Ts..., Back...>;
    };

    template <template <class...> typename F, typename... Back>
    struct bind_back1_func
    {
        template <typename A>
        using apply = F<A, Back...>;
    };

    template <template <class...> typename F, typename... Back>
    struct bind_back2_func
    {
        template <typename A, typename B>
        using apply = F<A, B, Back...>;
    };

    template <template <class...> typename F, typename... Back>
    struct bind_back3_func
    {
        template <typename A, typename B, typename C>
        using apply = F<A, B, C, Back...>;
    };

    template <typename F, typename... Back>
    using bind_back = bind_back_func<F::template apply, Back...>;

    template <typename F, typename... Back>
    using bind_back1 = bind_back1_func<F::template apply, Back...>;

    template <typename F, typename... Back>
    using bind_back2 = bind_back2_func<F::template apply, Back...>;

    template <typename F, typename... Back>
    using bind_back3 = bind_back3_func<F::template apply, Back...>;

    ///////////////////////////////////////////////////////////////////////////
    namespace detail {

        template <bool>
        struct if_
        {
            template <typename TrueCase, typename>
            using apply = TrueCase;
        };

        template <>
        struct if_<false>
        {
            template <typename, typename FalseCase>
            using apply = FalseCase;
        };
    }    // namespace detail

    template <typename Cond, typename TrueCase, typename FalseCase>
    using if_ = invoke2<detail::if_<value<Cond>>, TrueCase, FalseCase>;

    ///////////////////////////////////////////////////////////////////////////
    namespace detail {

        template <typename... Ts>
        struct compose_args_helper_not_valid;

        template <typename F, typename Pack, typename Enable = void>
        struct compose_args_helper
        {
            using type = compose_args_helper_not_valid<F, Pack>;
        };

        template <typename F, template <class...> typename A, typename... As>
        struct compose_args_helper<F, pack<A<As...>>,
            std::enable_if_t<value<is_invocable<F, As...>>>>
        {
            using type = invoke<F, As...>;
        };

        template <typename F, template <class...> typename A, typename... As,
            template <typename...> typename B, typename... Bs, typename... Rest>
        struct compose_args_helper<F, pack<A<As...>, B<Bs...>, Rest...>>
          : compose_args_helper<F, pack<pack<As..., Bs...>, Rest...>>
        {
        };

        template <typename F, template <class...> typename A, typename... As,
            template <class...> typename B, typename... Bs,
            template <class...> typename C, typename... Cs, typename... Rest>
        struct compose_args_helper<F,
            pack<A<As...>, B<Bs...>, C<Cs...>, Rest...>>
          : compose_args_helper<F, pack<pack<As..., Bs..., Cs...>, Rest...>>
        {
        };

        template <typename F, template <class...> typename A, typename... As,
            template <class...> typename B, typename... Bs,
            template <class...> typename C, typename... Cs,
            template <class...> typename D, typename... Ds, typename... Rest>
        struct compose_args_helper<F,
            pack<A<As...>, B<Bs...>, C<Cs...>, D<Ds...>, Rest...>>
          : compose_args_helper<F,
                pack<pack<As..., Bs..., Cs..., Ds...>, Rest...>>
        {
        };

        template <typename F>
        struct compose_args_helper<F, pack<>>
          : compose_args_helper<F, pack<pack<>>>
        {
        };
    }    // namespace detail

    template <typename F = func<pack>>
    struct compose_args
    {
        template <typename... Ts>
        using apply = type<detail::compose_args_helper<F, pack<Ts...>>>;
    };

    namespace detail {

        template <template <class...> typename F, typename Pack,
            typename Enable = void>
        struct defer_helper
        {
        };

        template <template <class...> typename F, typename... Ts>
        struct defer_helper<F, pack<Ts...>,
            std::enable_if_t<value<is_valid<F, Ts...>>>>
        {
            using type = F<Ts...>;
        };
    }    // namespace detail

    template <template <class...> typename F>
    struct defer
    {
        template <typename... Ts>
        using apply = type<detail::defer_helper<F, pack<Ts...>>>;
    };

    ///////////////////////////////////////////////////////////////////////////
    template <bool Value>
    using bool_ = std::bool_constant<Value>;

    template <typename Bool>
    using not_ = bool_<!value<Bool>>;

    template <typename T0, typename T1>
    using or_ = bool_<(value<T0> || value<T1>)>;

    template <typename T0, typename T1>
    using and_ = bool_<(value<T0> && value<T1>)>;

    ///////////////////////////////////////////////////////////////////////////
    template <typename T>
    struct constant
    {
        template <typename...>
        using apply = T;
    };

    struct count
    {
        template <typename... Ts>
        using apply = std::integral_constant<std::size_t, sizeof...(Ts)>;
    };

    template <typename F>
    struct curry
    {
        template <typename... Ts>
        using apply = invoke<F, Ts...>;
    };

    template <typename F = func<pack>>
    using uncurry = compose_args<F>;

    template <typename F, typename... List>
    using apply = invoke<uncurry<F>, List...>;

    template <typename F, typename Continuation = func<pack>>
    struct transform
    {
        template <typename... Ts>
        using apply = invoke<Continuation, invoke1<F, Ts>...>;
    };

    template <typename Pred = func<std::is_same>>
    struct not_pred
    {
        template <typename T1, typename T2>
        using apply = not_<invoke<Pred, T1, T2>>;
    };

    template <typename T, typename Cmp = func<std::is_same>>
    struct contains
    {
        template <typename... Ts>
        using apply = invoke<func<util::any_of>, invoke2<Cmp, T, Ts>...>;
    };

    template <typename T, typename... Ts>
    using one_of = invoke<contains<T>, Ts...>;

    template <typename T, typename... Ts>
    using none_of = invoke<contains<T, not_pred<func<std::is_same>>>, Ts...>;

    template <typename Continuation = func<pack>>
    struct push_back
    {
        template <typename List, typename... Item>
        using apply = meta::apply<bind_back<Continuation, Item...>, List>;
    };

    // clang-format off
    template <typename Continuation = func<pack>>
    struct push_back_unique
    {
        template <typename List, typename Item>
        using apply = meta::apply<
            if_<meta::apply<contains<Item>, List>,
                Continuation,
                bind_back<Continuation, Item>>,
            List>;
    };
    // clang-format on

    namespace detail {

        template <typename F, typename Pack, typename Enable = void>
        struct right_fold_helper
        {
        };

        template <typename F, typename State, typename Head, typename... Tail>
        struct right_fold_helper<F, pack<State, Head, Tail...>,
            std::enable_if_t<value<is_invocable2<F, State, Head>>>>
          : right_fold_helper<F, pack<invoke2<F, State, Head>, Tail...>>
        {
        };

        template <typename F, typename State>
        struct right_fold_helper<F, pack<State>>
        {
            using type = State;
        };
    }    // namespace detail

    template <typename Init, typename F>
    struct right_fold
    {
        template <typename... Ts>
        using apply = type<detail::right_fold_helper<F, pack<Init, Ts...>>>;
    };

    template <typename Continuation = func<pack>>
    struct unique
    {
        template <typename... Ts>
        using apply = meta::apply<Continuation,
            invoke<right_fold<pack<>, push_back_unique<>>, Ts...>>;
    };

    template <typename Old, typename New, typename Continuation = func<pack>>
    struct replace
    {
        template <typename... Ts>
        using apply =
            invoke<Continuation, if_<std::is_same<Ts, Old>, New, Ts>...>;
    };

    template <typename Old, typename Continuation = func<pack>>
    struct remove
    {
        template <typename... Ts>
        using apply = invoke<compose_args<Continuation>,
            if_<std::is_same<Ts, Old>, pack<>, pack<Ts>>...>;
    };

    template <typename A, typename... As>
    struct front
    {
        using type = A;
    };

    template <template <typename...> typename Fn>
    struct compose_template_func
    {
        template <typename... Args>
        using apply = Fn<Args...>;
    };

    template <typename... As>
    using single_t =
        std::enable_if_t<sizeof...(As) == 1, meta::type<front<As...>>>;

    template <typename Ty>
    struct single_or
    {
        template <typename... As>
        using apply =
            std::enable_if_t<sizeof...(As) <= 1, meta::type<front<As..., Ty>>>;
    };

    template <typename T, typename = void>
    inline constexpr bool has_id_v = false;

    template <typename T>
    inline constexpr bool has_id_v<T, std::void_t<typename T::id>> = true;

    template <typename T>
    struct has_id : std::integral_constant<bool, has_id_v<T>>
    {
    };

    template <bool val = true>
    struct get_id_func
    {
        template <typename T>
        using apply = typename T::id;
    };

    template <>
    struct get_id_func<false>
    {
        template <typename T>
        using apply = type_identity<T>;
    };

    template <typename T>
    using get_id_t = hpx::type_identity_t<
        hpx::meta::invoke<get_id_func<value<has_id<T>>>, T>>;

    template <typename T, typename... As>
    inline constexpr bool is_constructible_from_v =
        std::is_destructible_v<T>&& std::is_constructible_v<T, As...>;

    template <typename T, typename... As>
    inline constexpr bool is_nothrow_constructible_from_v =
        is_constructible_from_v<T, As...>&&
            std::is_nothrow_constructible_v<T, As...>;

}    // namespace hpx::meta
