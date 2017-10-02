//  Copyright (c) 2007-2017 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/executors/thread_execution_parameters.hpp

#if !defined(HPX_PARALLEL_THREAD_EXECUTOR_PARAMETER_TRAITS_AUG_26_2015_1204PM)
#define HPX_PARALLEL_THREAD_EXECUTOR_PARAMETER_TRAITS_AUG_26_2015_1204PM

#include <hpx/config.hpp>
#include <hpx/runtime/threads/thread_executor.hpp>
#include <hpx/traits/is_executor_parameters.hpp>

#include <hpx/parallel/executors/execution_parameters.hpp>

#include <cstddef>
#include <type_traits>

namespace hpx { namespace threads
{
    // reset_thread_distribution dispatch point
    template <typename Parameters, typename Executor>
    HPX_FORCEINLINE typename std::enable_if<
        hpx::traits::is_threads_executor<Executor>::value
    >::type
    reset_thread_distribution(Parameters && params, Executor && sched)
    {
        sched.reset_thread_distribution();
    }
}}

#endif
