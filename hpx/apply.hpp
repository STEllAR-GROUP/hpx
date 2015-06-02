//  Copyright (c) 2007-2014 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_APPLY_APR_16_20012_0943AM)
#define HPX_APPLY_APR_16_20012_0943AM

#include <hpx/hpx_fwd.hpp>
#include <hpx/async.hpp>
#include <hpx/runtime/threads/thread_helpers.hpp>
#include <hpx/runtime/threads/thread_executor.hpp>
#include <hpx/runtime/applier/apply.hpp>
#include <hpx/runtime/applier/apply_continue.hpp>
#include <hpx/util/bind_action.hpp>
#include <hpx/util/decay.hpp>
#include <hpx/util/deferred_call.hpp>
#include <hpx/util/move.hpp>
#include <hpx/traits/is_action.hpp>
#include <hpx/traits/is_callable.hpp>
#include <hpx/traits/is_executor.hpp>

#include <boost/utility/enable_if.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx
{
    ///////////////////////////////////////////////////////////////////////////
    // Define apply() overloads for plain local functions and function objects.

    // Simply launch the given function or function object asynchronously
    template <typename F, typename ...Ts>
    typename boost::enable_if_c<
        traits::detail::is_deferred_callable<F(Ts...)>::value
      , bool
    >::type
    apply(threads::executor& sched, F&& f, Ts&&... vs)
    {
        sched.add(
            util::deferred_call(std::forward<F>(f), std::forward<Ts>(vs)...),
            "hpx::apply");
        return false;
    }

    template <typename Executor, typename F, typename ...Ts>
    typename boost::enable_if_c<
        boost::enable_if_c<
            traits::is_executor<typename util::decay<Executor>::type>::value
          , traits::detail::is_deferred_callable<F(Ts...)>
        >::type::value
      , bool
    >::type
    apply(Executor& exec, F&& f, Ts&&... vs)
    {
        parallel::executor_traits<Executor>::apply_execute(exec,
            util::deferred_call(std::forward<F>(f), std::forward<Ts>(vs)...));
        return false;
    }

    template <typename F, typename ...Ts>
    typename boost::enable_if_c<
        boost::enable_if_c<
            !traits::is_executor<typename util::decay<F>::type>::value
         && !traits::is_action<typename util::decay<F>::type>::value
         && !traits::is_bound_action<typename util::decay<F>::type>::value
          , traits::detail::is_deferred_callable<F(Ts...)>
        >::type::value
      , bool
    >::type
    apply(F&& f, Ts&&... vs)
    {
        threads::register_thread_nullary(
            util::deferred_call(std::forward<F>(f), std::forward<Ts>(vs)...),
            "hpx::apply");
        return false;
    }

    ///////////////////////////////////////////////////////////////////////////
    // Define apply() overloads for bound actions
    template <typename Action, typename BoundArgs, typename ...Ts>
    bool apply(
        hpx::util::detail::bound_action<Action, BoundArgs> const& bound
      , Ts&&... vs
    )
    {
        return bound.apply(std::forward<Ts>(vs)...);
    }
}

#endif
