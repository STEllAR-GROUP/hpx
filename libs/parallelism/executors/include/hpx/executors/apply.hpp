//  Copyright (c) 2007-2015 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/async_base/apply.hpp>
#include <hpx/execution/executors/execution.hpp>
#include <hpx/execution/traits/is_executor.hpp>
#include <hpx/executors/parallel_executor.hpp>
#include <hpx/functional/deferred_call.hpp>

#include <type_traits>
#include <utility>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace detail {
    ///////////////////////////////////////////////////////////////////////////
    // Define apply() overloads for plain local functions and function objects.
    // dispatching trait for hpx::apply

    // launch a plain function/function object
    template <typename Func, typename Enable>
    struct apply_dispatch
    {
        template <typename F, typename... Ts>
        HPX_FORCEINLINE static typename std::enable_if<
            traits::detail::is_deferred_invocable<F, Ts...>::value, bool>::type
        call(F&& f, Ts&&... ts)
        {
            execution::parallel_executor exec;
            exec.post(std::forward<F>(f), std::forward<Ts>(ts)...);
            return false;
        }
    };

    // The overload for hpx::apply taking an executor simply forwards to the
    // corresponding executor customization point.
    //
    // parallel::execution::executor
    // threads::executor
    template <typename Executor>
    struct apply_dispatch<Executor,
        typename std::enable_if<traits::is_one_way_executor<Executor>::value ||
            traits::is_two_way_executor<Executor>::value
#if defined(HPX_HAVE_THREAD_EXECUTORS_COMPATIBILITY)
            || traits::is_threads_executor<Executor>::value
#endif
            >::type>
    {
        template <typename Executor_, typename F, typename... Ts>
        HPX_FORCEINLINE static decltype(auto) call(
            Executor_&& exec, F&& f, Ts&&... ts)
        {
            parallel::execution::post(std::forward<Executor_>(exec),
                std::forward<F>(f), std::forward<Ts>(ts)...);
            return false;
        }
    };
}}    // namespace hpx::detail
