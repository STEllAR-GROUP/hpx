//  Copyright (c)      2013 Thomas Heller
//  Copyright (c) 2007-2015 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_LCOS_LOCAL_DATAFLOW_HPP
#define HPX_LCOS_LOCAL_DATAFLOW_HPP

#include <hpx/config.hpp>
#include <hpx/traits/acquire_future.hpp>
#include <hpx/traits/future_access.hpp>
#include <hpx/traits/is_action.hpp>
#include <hpx/traits/is_executor.hpp>
#include <hpx/lcos/future.hpp>
#include <hpx/lcos/dataflow.hpp>
#include <hpx/runtime/launch_policy.hpp>
#include <hpx/runtime/threads/thread_executor.hpp>
#include <hpx/util/tuple.hpp>

#include <boost/intrusive_ptr.hpp>

#include <type_traits>
#include <utility>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace lcos { namespace detail
{
    template <typename Action>
    struct dataflow_launch_policy_dispatch<Action,
        typename std::enable_if<!traits::is_action<Action>::value>::type>
    {
        template <typename F, typename ...Ts>
        HPX_FORCEINLINE static
        typename dataflow_frame<
            launch
          , typename std::decay<F>::type
          , util::tuple<
                typename traits::acquire_future<Ts>::type...
            >
        >::type
        call(launch policy, F && f, Ts &&... ts)
        {
            typedef
                dataflow_frame<
                    launch
                  , typename std::decay<F>::type
                  , util::tuple<
                        typename traits::acquire_future<Ts>::type...
                    >
                >
                frame_type;

            boost::intrusive_ptr<frame_type> p(new frame_type(
                    policy
                  , std::forward<F>(f)
                  , util::forward_as_tuple(
                        traits::acquire_future_disp()(
                            std::forward<Ts>(ts)
                        )...
                    )
                ));
            p->do_await();

            using traits::future_access;
            return future_access<typename frame_type::type>::create(std::move(p));
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    // launch
    template <typename Policy>
    struct dataflow_dispatch<Policy,
        typename std::enable_if<traits::is_launch_policy<Policy>::value>::type>
    {
        template <typename F, typename ...Ts>
        HPX_FORCEINLINE static auto
        call(launch const& policy, F && f, Ts &&... ts)
        ->  decltype(dataflow_launch_policy_dispatch<
                    typename std::decay<F>::type
                >::call(policy, std::forward<F>(f), std::forward<Ts>(ts)...))
        {
            return dataflow_launch_policy_dispatch<
                    typename std::decay<F>::type
                >::call(policy, std::forward<F>(f), std::forward<Ts>(ts)...);
        }
    };

    // plain function or function object
    template <typename Func, typename Enable>
    struct dataflow_dispatch
    {
        template <typename F, typename ...Ts>
        HPX_FORCEINLINE static
        typename detail::dataflow_frame<
            launch
          , typename std::decay<F>::type
          , util::tuple<
                typename traits::acquire_future<Ts>::type...
            >
        >::type
        call(F && f, Ts &&... ts)
        {
            return dataflow_dispatch<launch>::call(
                launch::all, std::forward<F>(f), std::forward<Ts>(ts)...);
        }
    };

    // threads::executor
    template <typename Executor>
    struct dataflow_dispatch<Executor,
        typename std::enable_if<traits::is_threads_executor<Executor>::value>::type>
    {
        template <typename F, typename ...Ts>
        HPX_FORCEINLINE static
        typename dataflow_frame<
            threads::executor
          , typename std::decay<F>::type
          , util::tuple<
                typename traits::acquire_future<Ts>::type...
            >
        >::type
        call(Executor& sched, F && f, Ts &&... ts)
        {
            typedef
                dataflow_frame<
                    threads::executor
                  , typename std::decay<F>::type
                  , util::tuple<
                        typename traits::acquire_future<Ts>::type...
                    >
                >
                frame_type;

            boost::intrusive_ptr<frame_type> p(new frame_type(
                    sched
                  , std::forward<F>(f)
                  , util::forward_as_tuple(
                        traits::acquire_future_disp()(
                            std::forward<Ts>(ts)
                        )...
                    )
                ));
            p->do_await();

            using traits::future_access;
            return future_access<typename frame_type::type>::create(std::move(p));
        }
    };

    // parallel executor
    template <typename Executor>
    struct dataflow_dispatch<Executor,
        typename std::enable_if<traits::is_executor<Executor>::value>::type>
    {
        template <typename F, typename ...Ts>
        HPX_FORCEINLINE static
        typename detail::dataflow_frame<
            Executor
          , typename std::decay<F>::type
          , util::tuple<
                typename traits::acquire_future<Ts>::type...
            >
        >::type
        call(Executor& exec, F && f, Ts &&... ts)
        {
            typedef
                detail::dataflow_frame<
                    Executor
                  , typename std::decay<F>::type
                  , util::tuple<
                        typename traits::acquire_future<Ts>::type...
                    >
                >
                frame_type;

            boost::intrusive_ptr<frame_type> p(new frame_type(
                    exec
                  , std::forward<F>(f)
                  , util::forward_as_tuple(
                        traits::acquire_future_disp()(std::forward<Ts>(ts))...
                    )
                ));
            p->do_await();

            using traits::future_access;
            return future_access<typename frame_type::type>::create(std::move(p));
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
        HPX_FORCEINLINE
        auto dataflow(F && f, Ts &&... ts)
        ->  decltype(lcos::detail::dataflow_dispatch<
                typename std::decay<F>::type>::call(
                    std::forward<F>(f), std::forward<Ts>(ts)...
                ))
        {
            return lcos::detail::dataflow_dispatch<
                typename std::decay<F>::type>::call(
                    std::forward<F>(f), std::forward<Ts>(ts)...);
        }
    }}
#endif

    template <typename F, typename ...Ts>
    HPX_FORCEINLINE
    auto dataflow(F && f, Ts &&... ts)
    ->  decltype(lcos::detail::dataflow_dispatch<
            typename std::decay<F>::type>::call(
                std::forward<F>(f), std::forward<Ts>(ts)...
            ))
    {
        return lcos::detail::dataflow_dispatch<
            typename std::decay<F>::type>::call(
                std::forward<F>(f), std::forward<Ts>(ts)...);
    }
}

#endif /*HPX_LCOS_LOCAL_DATAFLOW_HPP*/
