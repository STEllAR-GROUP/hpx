//  Copyright (c)      2013 Thomas Heller
//  Copyright (c) 2007-2015 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_LCOS_LOCAL_DATAFLOW_HPP
#define HPX_LCOS_LOCAL_DATAFLOW_HPP

#include <hpx/config.hpp>
#include <hpx/lcos/dataflow.hpp>
#include <hpx/lcos/future.hpp>
#include <hpx/runtime/launch_policy.hpp>
#include <hpx/runtime/threads/thread_executor.hpp>
#include <hpx/traits/acquire_future.hpp>
#include <hpx/traits/future_access.hpp>
#include <hpx/traits/is_action.hpp>
#include <hpx/traits/v1/is_executor.hpp>
#include <hpx/traits/is_launch_policy.hpp>
#include <hpx/util/tuple.hpp>

#include <boost/intrusive_ptr.hpp>

#include <type_traits>
#include <utility>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace lcos { namespace detail
{
    template <typename F, typename Enable = void>
    struct dataflow_dispatch;

    // launch
    template <typename Policy>
    struct dataflow_dispatch<Policy,
        typename std::enable_if<
            traits::is_launch_policy<typename std::decay<Policy>::type>::value
        >::type>
    {
        template <typename F, typename ...Ts>
        HPX_FORCEINLINE static typename std::enable_if<
            !traits::is_action<typename std::decay<F>::type>::value,
            lcos::future<
                typename detail::dataflow_return<
                    typename std::decay<F>::type,
                    util::tuple<typename traits::acquire_future<Ts>::type...>
                >::type>
        >::type
        call(Policy && policy, F && f, Ts &&... ts)
        {
            typedef dataflow_frame<
                typename std::decay<Policy>::type,
                typename std::decay<F>::type,
                util::tuple<typename traits::acquire_future<Ts>::type...>
            > frame_type;

            // Create the data which is used to construct the dataflow_frame
            auto data = frame_type::construct_from(
                std::forward<Policy>(policy), std::forward<F>(f));

            // Construct the dataflow_frame and traverse
            // the arguments asynchronously
            boost::intrusive_ptr<frame_type> p = util::traverse_pack_async(
                util::async_traverse_in_place_tag<frame_type>{},
                std::move(data),
                traits::acquire_future_disp()(std::forward<Ts>(ts))...);

            using traits::future_access;
            return future_access<typename frame_type::type>::create(std::move(p));
        }
    };

    // threads::executor
    template <typename Executor>
    struct dataflow_dispatch<Executor,
        typename std::enable_if<
            traits::is_threads_executor<typename std::decay<Executor>::type>::value
        >::type>
    {
        template <typename F, typename ...Ts>
        HPX_FORCEINLINE static typename std::enable_if<
            !traits::is_action<typename std::decay<F>::type>::value,
            lcos::future<
                typename detail::dataflow_return<
                    typename std::decay<F>::type,
                    util::tuple<typename traits::acquire_future<Ts>::type...>
                >::type>
        >::type
        call(Executor& sched, F && f, Ts &&... ts)
        {
            typedef dataflow_frame<
                threads::executor,
                typename std::decay<F>::type,
                util::tuple<typename traits::acquire_future<Ts>::type...>
            > frame_type;

            // Create the data which is used to construct the dataflow_frame
            auto data = frame_type::construct_from(sched, std::forward<F>(f));

            // Construct the dataflow_frame and traverse
            // the arguments asynchronously
            boost::intrusive_ptr<frame_type> p = util::traverse_pack_async(
                util::async_traverse_in_place_tag<frame_type>{},
                std::move(data),
                traits::acquire_future_disp()(std::forward<Ts>(ts))...);

            using traits::future_access;
            return future_access<typename frame_type::type>::create(std::move(p));
        }
    };

    // parallel executors
    template <typename Executor>
    struct dataflow_dispatch<Executor,
        typename std::enable_if<
#if defined(HPX_HAVE_EXECUTOR_COMPATIBILITY)
                traits::is_executor<
                    typename std::decay<Executor>::type>::value ||
#endif
                traits::is_one_way_executor<
                    typename std::decay<Executor>::type>::value ||
                traits::is_two_way_executor<
                    typename std::decay<Executor>::type>::value
        >::type>
    {
        template <typename F, typename ...Ts>
        HPX_FORCEINLINE static typename std::enable_if<
            !traits::is_action<typename std::decay<F>::type>::value,
            lcos::future<
                typename detail::dataflow_return<
                    typename std::decay<F>::type,
                    util::tuple<typename traits::acquire_future<Ts>::type...>
                >::type>
        >::type
        call(Executor && exec, F && f, Ts &&... ts)
        {
            typedef dataflow_frame<
                typename std::decay<Executor>::type,
                typename std::decay<F>::type,
                util::tuple<typename traits::acquire_future<Ts>::type...>
            > frame_type;

            // Create the data which is used to construct the dataflow_frame
            auto data = frame_type::construct_from(
                std::forward<Executor>(exec), std::forward<F>(f));

            // Construct the dataflow_frame and traverse
            // the arguments asynchronously
            boost::intrusive_ptr<frame_type> p = util::traverse_pack_async(
                util::async_traverse_in_place_tag<frame_type>{},
                std::move(data),
                traits::acquire_future_disp()(std::forward<Ts>(ts))...);

            using traits::future_access;
            return future_access<typename frame_type::type>::create(std::move(p));
        }
    };

    // plain function or function object
    template <typename F>
    struct dataflow_dispatch<F,
        typename std::enable_if<
            !traits::is_action<typename std::decay<F>::type>::value &&
            !traits::is_launch_policy<typename std::decay<F>::type>::value &&
            !traits::is_threads_executor<typename std::decay<F>::type>::value &&
            !(
#if defined(HPX_HAVE_EXECUTOR_COMPATIBILITY)
                traits::is_executor<typename std::decay<F>::type>::value ||
#endif
                traits::is_one_way_executor<typename std::decay<F>::type>::value ||
                traits::is_two_way_executor<typename std::decay<F>::type>::value)
        >::type>
    {
        template <typename F_, typename ...Ts>
        HPX_FORCEINLINE static auto
        call(F_ && f, Ts &&... ts)
        ->  decltype(dataflow_dispatch<launch>::call(
                launch::async, std::forward<F_>(f), std::forward<Ts>(ts)...))
        {
            return dataflow_dispatch<launch>::call(
                launch::async, std::forward<F_>(f), std::forward<Ts>(ts)...);
        }
    };
}}}

///////////////////////////////////////////////////////////////////////////////
// local dataflow: invokes given function (or executor) when ready
namespace hpx
{
#if defined(HPX_HAVE_LOCAL_DATAFLOW_COMPATIBILITY)
    namespace lcos { namespace local
    {
        template <typename F, typename ...Ts>
        HPX_DEPRECATED(HPX_DEPRECATED_MSG) HPX_FORCEINLINE
        auto dataflow(F && f, Ts &&... ts)
        ->  decltype(lcos::detail::dataflow_dispatch<F>::call(
                std::forward<F>(f), std::forward<Ts>(ts)...))
        {
            return lcos::detail::dataflow_dispatch<F>::call(
                std::forward<F>(f), std::forward<Ts>(ts)...);
        }
    }}
#endif

    template <typename F, typename ...Ts>
    HPX_FORCEINLINE
    auto dataflow(F && f, Ts &&... ts)
    ->  decltype(lcos::detail::dataflow_dispatch<F>::call(
            std::forward<F>(f), std::forward<Ts>(ts)...))
    {
        return lcos::detail::dataflow_dispatch<F>::call(
            std::forward<F>(f), std::forward<Ts>(ts)...);
    }
}

#endif /*HPX_LCOS_LOCAL_DATAFLOW_HPP*/
