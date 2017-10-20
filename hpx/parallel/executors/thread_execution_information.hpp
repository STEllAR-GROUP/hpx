//  Copyright (c) 2017 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PARALLEL_EXECUTORS_THREAD_EXECUTION_INFORMATION_JAN_17_2017_0130PM)
#define HPX_PARALLEL_EXECUTORS_THREAD_EXECUTION_INFORMATION_JAN_17_2017_0130PM

#include <hpx/config.hpp>
#include <hpx/runtime/get_os_thread_count.hpp>
#include <hpx/runtime/threads/policies/scheduler_mode.hpp>
#include <hpx/runtime/threads/policies/topology.hpp>
#include <hpx/runtime/threads/thread_executor.hpp>
#include <hpx/runtime/threads/topology.hpp>
#include <hpx/traits/is_launch_policy.hpp>

#include <hpx/parallel/executors/execution_information.hpp>

#include <cstddef>
#include <type_traits>
#include <utility>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace threads
{
    template <typename Executor, typename Parameters>
    typename std::enable_if<
        hpx::traits::is_threads_executor<Executor>::value,
        std::size_t
    >::type
    processing_units_count(Executor && exec, Parameters&)
    {
        return hpx::get_os_thread_count(exec);
    }

    template <typename Executor>
    typename std::enable_if<
        hpx::traits::is_threads_executor<Executor>::value,
        bool
    >::type
    has_pending_closures(Executor && exec)
    {
        return exec.num_pending_closures() ? true : false;
    }

    template <typename Executor>
    typename std::enable_if<
        hpx::traits::is_threads_executor<Executor>::value,
        threads::mask_cref_type
    >::type
    get_pu_mask(Executor && exec, threads::topology& topo,
        std::size_t thread_num)
    {
        return exec.get_pu_mask(topo, thread_num);
    }

    template <typename Executor, typename Mode>
    typename std::enable_if<
        hpx::traits::is_threads_executor<Executor>::value
    >::type
    set_scheduler_mode(Executor && exec, Mode mode)
    {
        exec.set_scheduler_mode(
            static_cast<threads::policies::scheduler_mode>(mode)
        );
    }
}}

#endif

