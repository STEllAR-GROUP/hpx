//  Copyright (c) 2022 Shreyas Atre
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <coroutine>
#include <type_traits>

namespace hpx { namespace execution { namespace experimental {

    template <typename Promise, typename Awaiter>
    decltype(auto) await_suspend(Awaiter&& await)
    {
        if constexpr (!std::is_same_v<Promise, void>)
        {
            return await.await_suspend(std::coroutine_handle<Promise>{});
        }
        return;
    }

    namespace detail {

        template <typename T, typename... Ts>
        bool inline constexpr one_of = (std::is_same_v<T, Ts> || ...);

        template <class, template <class...> class>
        constexpr bool is_instance_of_ = false;

        template <class... _As, template <class...> class _T>
        constexpr bool is_instance_of_<_T<_As...>, _T> = true;

        template <class _Ty, template <class...> class _T>
        constexpr bool is_instance_of = is_instance_of_<_Ty, _T>;

        template <typename T>
        inline constexpr bool is_await_suspend_result_t =
            one_of<T, void, bool> || is_instance_of<T, std::coroutine_handle>;

        template <typename, typename = void>
        constexpr bool has_await_ready{};

        template <typename T>
        constexpr bool has_await_ready<T,
            std::void_t<decltype(std::declval<T>().await_ready())>> = true;

        template <typename, typename = void>
        constexpr bool has_await_resume{};

        template <typename T>
        constexpr bool has_await_resume<T,
            std::void_t<decltype(std::declval<T>().await_resume())>> = true;

        template <typename, typename = void>
        constexpr bool has_await_suspend{};

        template <typename T>
        constexpr bool has_await_suspend<T,
            std::void_t<decltype(std::declval<T>().await_suspend())>> = true;

        template <typename, typename, typename = void>
        constexpr bool has_await_suspend_coro_handle{};

        template <typename T, typename Ts>
        constexpr bool has_await_suspend_coro_handle<T, Ts,
            std::void_t<decltype(std::declval<T>().await_suspend(
                std::coroutine_handle<Ts>{}))>> = true;

        template <bool await_ready, typename Awaiter, typename Promise>
        struct is_awaiter_impl;

        template <typename Awaiter, typename Promise>
        struct is_awaiter_impl<false, Awaiter, Promise> : std::false_type
        {
        };

        template <typename Awaiter, typename Promise>
        struct is_awaiter_impl<true, Awaiter, Promise>
          : std::integral_constant<bool,
                is_await_suspend_result_t<decltype(
                    await_suspend<Promise>(std::declval<Awaiter>()))>>
        {
        };

    }    // namespace detail

    // An Awaiter type is a type that implements the three special methods
    // that are called as part of a co_await expression: await_ready,
    // await_suspend and await_resume.

    // https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2018/p1288r0.pdf
    // Lewis Baker - The rationale here is that for an awaiter object
    // to be able to support being awaited in an arbitrary natural coroutine
    // context it will generally need to type-erase the
    // coroutine_handle<Promise> to coroutine_handle<void> so that it can
    // store the continuation for an arbitrary coroutine-type.
    // If theawait_suspend() method overload-set only has overloads that
    // accept specific types of coroutine_handle<P> then it is only awaitable
    // within specific contexts and thus we don’t consider it to satisfy the
    // Awaiter concept.

    template <typename Awaiter, typename Promise = void>
    struct is_awaiter
      : detail::is_awaiter_impl<detail::has_await_ready<Awaiter> &&
                detail::has_await_resume<Awaiter> &&
                (detail::has_await_suspend<Awaiter> ||
                    detail::has_await_suspend_coro_handle<Awaiter, Promise>),
            Awaiter, Promise>
    {
    };

    template <typename Awaiter, typename Promise = void>
    bool inline constexpr is_awaiter_v = is_awaiter<Awaiter, Promise>::value;

    namespace detail {

        template <typename Awaitable, typename = void>
        constexpr bool has_member_operator_co_await_v = false;

        template <typename Awaitable>
        constexpr bool has_member_operator_co_await_v<Awaitable,
            std::void_t<decltype(
                std::declval<Awaitable>().operator co_await())>> = true;

        template <typename Awaitable, typename = void>
        constexpr bool has_free_operator_co_await_v = false;

        template <typename Awaitable>
        constexpr bool has_free_operator_co_await_v<Awaitable,
            std::void_t<decltype(operator co_await(
                std::declval<Awaitable>()))>> = true;

        template <typename, typename = void, typename...>
        constexpr bool has_await_transform{};

        template <typename Promise, typename... Ts>
        constexpr bool has_await_transform<Promise,
            std::void_t<decltype(std::declval<Promise>().await_transform(
                ((Ts &&) std::declval<Ts>())...))>,
            Ts...> = true;

    }    // namespace detail

    // Returns the result of applying operator co_await() to the function’s
    // argument, if the operator is defined, otherwise returns a reference
    // to the input argument.
    template <typename Awaitable>
    decltype(auto) get_awaiter(Awaitable&& await, void*)
    {
        if constexpr (detail::has_member_operator_co_await_v<Awaitable>)
        {
            return ((Awaitable &&) await).operator co_await();
        }
        else if constexpr (detail::has_free_operator_co_await_v<Awaitable>)
        {
            return operator co_await((Awaitable &&) await);
        }
        else
        {
            return (Awaitable &&) await;
        }
    }

    template <typename Awaitable, typename Promise,
        typename = std::enable_if_t<
            detail::has_await_transform<Promise, Awaitable>, void>>
    decltype(auto) get_awaiter(Awaitable&& await, Promise* promise)
    {
        if constexpr (detail::has_member_operator_co_await_v<decltype(
                          promise->await_transform((Awaitable &&) await))>)
        {
            return promise->await_transform((Awaitable &&) await)
                .
                operator co_await();
        }
        else if constexpr (detail::has_free_operator_co_await_v<decltype(
                               promise->await_transform((Awaitable &&) await))>)
        {
            return operator co_await(
                promise->await_transform((Awaitable &&) await));
        }
        else
        {
            return promise->await_transform((Awaitable &&) await);
        }
    }

    // Awaitable - Something that you can apply the ‘co_await’ operator to.
    // If the promise type defines an await_transform() member then
    // the awaitable is obtained by calling promise.await_transform(value),
    // passing the awaited value.
    // Otherwise, if the promise type does not define an await_transform()
    // member then the awaitable is the awaited value itself.
    namespace detail {
        template <bool hasAwaitSuspend, typename Awaitable, typename Promise>
        struct is_awaitable_impl;

        template <typename Awaitable, typename Promise>
        struct is_awaitable_impl<false, Awaitable, Promise> : std::false_type
        {
        };

        template <typename Awaitable, typename Promise>
        struct is_awaitable_impl<true, Awaitable, Promise>
          : std::integral_constant<bool,
                is_await_suspend_result_t<decltype(get_awaiter(
                    std::declval<Awaitable>(), std::declval<Promise>()))>>
        {
        };
    }    // namespace detail

    template <typename Awaitable, typename Promise>
    struct is_awaitable
      : detail::is_awaitable_impl<
            detail::has_await_suspend<Awaitable,
                decltype(std::coroutine_handle<Promise>{})>,
            Awaitable, Promise>
    {
    };

    template <typename Awaitable, typename Promise = void>
    inline constexpr bool is_awaitable_v =
        is_awaitable<Awaitable, Promise>::value;

    template <typename Awaitable, typename Promise = void,
        typename = std::enable_if_t<is_awaitable_v<Awaitable, Promise>>>
    using await_result_t =
        decltype((get_awaiter(std::declval<Awaitable>(), (Promise*) nullptr)
                      .await_resume()));

}}}    // namespace hpx::execution::experimental
