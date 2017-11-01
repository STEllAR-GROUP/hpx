//  Copyright (c) 2007-2017 Hartmut Kaiser
//  Copyright (c) 2013 Agustin Berge
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_TRAITS_FUTURE_THEN_RESULT_DEC_25_2016_1141AM)
#define HPX_TRAITS_FUTURE_THEN_RESULT_DEC_25_2016_1141AM

#include <hpx/config.hpp>
#include <hpx/util/always_void.hpp>
#include <hpx/util/detail/pack.hpp>
#include <hpx/util/identity.hpp>
#include <hpx/util/lazy_conditional.hpp>
#include <hpx/util/result_of.hpp>
#include <hpx/traits/is_executor.hpp>
#include <hpx/traits/is_future.hpp>
#include <hpx/traits/future_traits.hpp>

#include <type_traits>
#include <utility>

namespace hpx { namespace traits
{
    ///////////////////////////////////////////////////////////////////////////
    namespace detail
    {
        struct no_executor {};

        template <typename Executor, typename T, typename Ts,
            typename Enable = void>
        struct executor_future;

        template <typename Executor, typename T, typename ...Ts>
        struct executor_future<Executor, T, hpx::util::detail::pack<Ts...>,
            typename std::enable_if<
                hpx::traits::is_two_way_executor<Executor>::value
            >::type>
        {
            using type = decltype(
                std::declval<Executor&&>().async_execute(
                    std::declval<T(*)(Ts...)>(), std::declval<Ts>()...));
        };

        template <typename Executor, typename T, typename Ts>
        struct executor_future<Executor, T, Ts,
            typename std::enable_if<
                hpx::traits::is_one_way_executor<Executor>::value &&
               !hpx::traits::is_two_way_executor<Executor>::value
            >::type>
        {
            using type = hpx::lcos::future<T>;
        };
    }

    template <typename Executor, typename T, typename ...Ts>
    struct executor_future
      : detail::executor_future<
            typename std::decay<Executor>::type,
            T, hpx::util::detail::pack<typename std::decay<Ts>::type...> >
    {};

    template <typename Executor, typename T, typename ...Ts>
    using executor_future_t =
        typename executor_future<Executor, T, Ts...>::type;

    ///////////////////////////////////////////////////////////////////////////
    namespace detail
    {
        template <typename Future, typename F>
        struct continuation_not_callable
        {
#if defined(HPX_HAVE_CXX14_RETURN_TYPE_DEDUCTION)
            static auto error(Future future, F& f)
            {
                f(std::move(future));
            }
#else
            static auto error(Future future, F& f)
             -> decltype(f(std::move(future)));
#endif

            using type = decltype(
                error(std::declval<Future>(), std::declval<F&>()));
        };

        ///////////////////////////////////////////////////////////////////////
        template <typename Future, typename F, typename Enable = void>
        struct future_then_result
        {
            typedef typename continuation_not_callable<Future, F>::type type;
        };

        template <typename Future, typename F>
        struct future_then_result<
            Future, F,
            typename hpx::util::always_void<
                typename hpx::util::invoke_result<F&, Future>::type
            >::type
        >
        {
            typedef typename hpx::util::invoke_result<F&, Future>::type
                cont_result;

            // perform unwrapping of future<future<R>>
            typedef typename util::lazy_conditional<
                    hpx::traits::detail::is_unique_future<cont_result>::value,
                    hpx::traits::future_traits<cont_result>,
                    hpx::util::identity<cont_result>
                >::type result_type;

            typedef hpx::lcos::future<result_type> type;
        };

        ///////////////////////////////////////////////////////////////////////
        template <typename Executor, typename Future, typename F,
            typename Enable = void>
        struct future_then_executor_result
        {
            typedef typename continuation_not_callable<Future, F>::type type;
        };

        template <typename Executor, typename Future, typename F>
        struct future_then_executor_result<
                Executor, Future, F,
                typename hpx::util::always_void<
                    typename hpx::util::invoke_result<F&, Future>::type
                >::type>
        {
            typedef typename hpx::util::invoke_result<F&, Future>::type
                func_result_type;

            typedef typename traits::executor_future<
                    Executor, func_result_type, Future
                >::type cont_result;

            // perform unwrapping of future<future<R>>
            typedef typename util::lazy_conditional<
                    hpx::traits::detail::is_unique_future<cont_result>::value,
                    hpx::traits::future_traits<cont_result>,
                    hpx::util::identity<cont_result>
                >::type result_type;

            typedef hpx::lcos::future<result_type> type;
        };
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename Future, typename F>
    struct future_then_result
      : detail::future_then_result<Future, F>
    {};

    template <typename Executor, typename Future, typename F>
    struct future_then_executor_result
      : detail::future_then_executor_result<
            typename std::decay<Executor>::type, Future, F>
    {};
}}

#endif
