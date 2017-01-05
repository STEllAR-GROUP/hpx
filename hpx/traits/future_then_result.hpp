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

        template <typename Executor, typename T, typename Pack,
            typename Enable = void>
        struct executor_future;

        template <typename Executor, typename T, typename ... Ts>
        struct executor_future<Executor, T, hpx::util::detail::pack<Ts...>,
            typename std::enable_if<
                hpx::traits::is_two_way_executor<Executor>::value
            >::type>
        {
            using type = decltype(
                std::declval<Executor&&>().async_execute(
                    std::declval<T(*)(Ts...)>(), std::declval<Ts>()...));
        };

        template <typename Executor, typename T, typename Pack>
        struct executor_future<Executor, T, Pack,
            typename std::enable_if<
                hpx::traits::is_one_way_executor<Executor>::value &&
               !hpx::traits::is_two_way_executor<Executor>::value
            >::type>
        {
            using type = hpx::lcos::future<T>;
        };
    }

    template <typename Executor, typename T, typename ... Ts>
    struct executor_future
      : detail::executor_future<
            typename std::decay<Executor>::type,
            T, hpx::util::detail::pack<typename std::decay<Ts>::type...> >
    {};

    template <typename Executor, typename T, typename ... Ts>
    using executor_future_t =
        typename executor_future<Executor, T, Ts...>::type;

    ///////////////////////////////////////////////////////////////////////////
    namespace detail
    {
        template <typename Future, typename F, typename Pack>
        struct continuation_not_callable;

        template <typename Future, typename F, typename ... Ts>
        struct continuation_not_callable<
            Future, F, hpx::util::detail::pack<Ts...> >
        {
            void error(Future future, F& f, Ts &&... ts)
            {
                f(future, std::forward<Ts>(ts)...);
            }

            ~continuation_not_callable()
            {
                error(std::declval<Future>(), std::declval<F&>(),
                    std::declval<Ts>()...);
            }
        };

        ///////////////////////////////////////////////////////////////////////
        template <typename Future, typename F, typename Pack,
            typename Enable = void>
        struct future_then_result
        {
            typedef continuation_not_callable<Future, F, Pack> type;
        };

        template <typename Future, typename F, typename ... Ts>
        struct future_then_result<
            Future, F, hpx::util::detail::pack<Ts...>
          , typename hpx::util::always_void<
                typename hpx::util::result_of<F(Future, Ts...)>::type
            >::type
        >
        {
            typedef typename hpx::util::result_of<F(Future, Ts...)>::type
                cont_result;

            // perform unwrapping of future<future<R>>
            typedef typename util::lazy_conditional<
                    hpx::traits::detail::is_unique_future<cont_result>::value
                  , hpx::traits::future_traits<cont_result>
                  , hpx::util::identity<cont_result>
                >::type result_type;

            typedef hpx::lcos::future<result_type> type;
        };

        ///////////////////////////////////////////////////////////////////////
        template <typename Executor, typename Future, typename F, typename Pack,
            typename Enable = void>
        struct future_then_executor_result
        {
            typedef continuation_not_callable<Future, F, Pack> type;
        };

        template <typename Executor, typename Future, typename F,
            typename ... Ts>
        struct future_then_executor_result<
                Executor, Future, F, hpx::util::detail::pack<Ts...>
              , typename hpx::util::always_void<
                    typename hpx::util::result_of<F(Future, Ts...)>::type
                >::type>
        {
            typedef typename hpx::util::result_of<F(Future, Ts...)>::type
                func_result_type;

            typedef typename traits::executor_future<
                    Executor, func_result_type, Future, Ts...
                >::type cont_result;

            // perform unwrapping of future<future<R>>
            typedef typename util::lazy_conditional<
                    hpx::traits::detail::is_unique_future<cont_result>::value
                  , hpx::traits::future_traits<cont_result>
                  , hpx::util::identity<cont_result>
                >::type result_type;

            typedef hpx::lcos::future<result_type> type;
        };
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename Future, typename F, typename ... Ts>
    struct future_then_result
      : detail::future_then_result<Future, F, hpx::util::detail::pack<Ts...> >
    {};

    template <typename Executor, typename Future, typename F, typename ... Ts>
    struct future_then_executor_result
      : detail::future_then_executor_result<
            typename std::decay<Executor>::type, Future, F,
            hpx::util::detail::pack<Ts...> >
    {};
}}

#endif
