//  Copyright (c) 2007-2019 Hartmut Kaiser
//  Copyright (c) 2013 Agustin Berge
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_LCOS_FUTURE_EXEC_MAR_06_2012_1059AM)
#define HPX_LCOS_FUTURE_EXEC_MAR_06_2012_1059AM

#include <hpx/config.hpp>
#include <hpx/allocator_support/allocator_deleter.hpp>
#include <hpx/allocator_support/internal_allocator.hpp>
#include <hpx/assertion.hpp>
#include <hpx/concepts/concepts.hpp>
#include <hpx/errors.hpp>
#include <hpx/functional/bind.hpp>
#include <hpx/functional/function.hpp>
#include <hpx/functional/invoke.hpp>
#include <hpx/functional/result_of.hpp>
#include <hpx/functional/traits/is_callable.hpp>
#include <hpx/lcos/detail/future_data.hpp>
#include <hpx/lcos/detail/future_traits.hpp>
#include <hpx/lcos/future.hpp>
#include <hpx/memory/intrusive_ptr.hpp>
#include <hpx/runtime/launch_policy.hpp>
#include <hpx/serialization/detail/polymorphic_nonintrusive_factory.hpp>
#include <hpx/timing/steady_clock.hpp>
#include <hpx/traits/acquire_shared_state.hpp>
#include <hpx/traits/future_access.hpp>
#include <hpx/traits/future_then_result.hpp>
#include <hpx/traits/future_traits.hpp>
#include <hpx/execution/traits/is_executor.hpp>
#include <hpx/traits/is_future.hpp>
#include <hpx/traits/is_launch_policy.hpp>
#include <hpx/type_support/always_void.hpp>
#include <hpx/type_support/decay.hpp>
#include <hpx/type_support/identity.hpp>
#include <hpx/type_support/lazy_enable_if.hpp>
#include <hpx/type_support/void_guard.hpp>
#include <hpx/util/serialize_exception.hpp>

#if defined(HPX_HAVE_AWAIT)
    #include <hpx/lcos/detail/future_await_traits.hpp>
#endif

#include <exception>
#include <iterator>
#include <memory>
#include <type_traits>
#include <utility>

/* Beginning of the file that has to go with executors */
namespace hpx { namespace lcos { namespace detail
{
    template <typename Executor, typename Future, typename F>
    inline typename hpx::traits::future_then_executor_result<
        Executor, typename std::decay<Future>::type, F
    >::type
    then_execute_helper(Executor &&, F &&, Future &&);

    ///////////////////////////////////////////////////////////////////////////
    // launch
    template <typename Future, typename Policy>
    struct future_then_dispatch<Future, Policy,
        typename std::enable_if<
            traits::is_launch_policy<Policy>::value
        >::type>
    {
        template <typename Policy_, typename F>
        HPX_FORCEINLINE
        static typename hpx::traits::future_then_result<Future, F>::type
        call(Future && fut, Policy_ && policy, F && f)
        {
            using result_type =
                typename hpx::traits::future_then_result<Future, F>::result_type;
            using continuation_result_type =
                typename hpx::util::invoke_result<F, Future>::type;

            typename hpx::traits::detail::shared_state_ptr<result_type>::type p =
                detail::make_continuation_alloc<continuation_result_type>(
                    hpx::util::internal_allocator<>{},
                    std::move(fut), std::forward<Policy_>(policy),
                    std::forward<F>(f));
            return hpx::traits::future_access<future<result_type> >::create(
                std::move(p));
        }

        template <typename Allocator, typename Policy_, typename F>
        HPX_FORCEINLINE static
        typename hpx::traits::future_then_result<Future, F>::type
        call_alloc(
            Allocator const& alloc, Future&& fut, Policy_&& policy, F&& f)
        {
            using result_type =
                typename hpx::traits::future_then_result<Future, F>::result_type;
            using continuation_result_type =
                typename hpx::util::invoke_result<F, Future>::type;

            typename hpx::traits::detail::shared_state_ptr<result_type>::type p =
                detail::make_continuation_alloc<continuation_result_type>(
                    alloc, std::move(fut), std::forward<Policy_>(policy),
                    std::forward<F>(f));
            return hpx::traits::future_access<future<result_type> >::create(
                std::move(p));
        }
    };

    // The overload for future::then taking an executor simply forwards to the
    // corresponding executor customization point.
    //
    // parallel executors v2
    // threads::executor
    template <typename Future, typename Executor>
    struct future_then_dispatch<Future, Executor,
        typename std::enable_if<
            traits::is_one_way_executor<Executor>::value ||
            traits::is_two_way_executor<Executor>::value ||
            traits::is_threads_executor<Executor>::value
        >::type>
    {
        template <typename Executor_, typename F>
        HPX_FORCEINLINE
        static typename hpx::traits::future_then_executor_result<
            Executor_, Future, F>::type
        call(Future && fut, Executor_ && exec, F && f)
        {
            // simply forward this to executor
            return detail::then_execute_helper(std::forward<Executor_>(exec),
                std::forward<F>(f), std::move(fut));
        }

        template <typename Allocator, typename Executor_, typename F>
        HPX_FORCEINLINE
        static typename hpx::traits::future_then_executor_result<
            Executor_, Future, F>::type
        call_alloc(Allocator const&, Future && fut, Executor_ && exec, F && f)
        {
            return call(std::forward<Future>(fut),
                std::forward<Executor_>(exec), std::forward<F>(f));
        }
    };

    // plain function, or function object
    template <typename Future, typename FD>
    struct future_then_dispatch<Future, FD, typename std::enable_if<
        !traits::is_launch_policy<FD>::value &&
        !(
            traits::is_one_way_executor<FD>::value ||
            traits::is_two_way_executor<FD>::value ||
            traits::is_threads_executor<FD>::value)
        >::type>
    {
        template <typename F>
        HPX_FORCEINLINE static auto
        call(Future && fut, F && f)
        ->  decltype(future_then_dispatch<Future, launch>::call(
                std::move(fut), launch::all, std::forward<F>(f)))
        {
            return future_then_dispatch<Future, launch>::call(
                std::move(fut), launch::all, std::forward<F>(f));
        }

        template <typename Allocator, typename F>
        HPX_FORCEINLINE static auto
        call_alloc(Allocator const& alloc, Future && fut, F && f)
        ->  decltype(future_then_dispatch<Future, launch>::call_alloc(
                alloc, std::move(fut), launch::all, std::forward<F>(f)))
        {
            return future_then_dispatch<Future, launch>::call_alloc(
                alloc, std::move(fut), launch::all, std::forward<F>(f));
        }
    };

}}} // namespace hpx::lcos::detail

#endif
