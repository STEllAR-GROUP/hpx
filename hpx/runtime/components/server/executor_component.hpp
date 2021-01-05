//  Copyright (c) 2017 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/async_base/traits/is_launch_policy.hpp>
#include <hpx/components_base/get_lva.hpp>
#include <hpx/coroutines/thread_enums.hpp>
#include <hpx/functional/deferred_call.hpp>
#include <hpx/naming_base/address.hpp>
#include <hpx/threading_base/annotated_function.hpp>
#include <hpx/threading_base/thread_description.hpp>
#include <hpx/threading_base/thread_init_data.hpp>

#include <hpx/execution/executors/execution.hpp>

#include <type_traits>
#include <utility>

namespace hpx { namespace components
{
    // This is a base class which allows to associate the execution of all
    // actions for a particular component instance with a given executor.
    template <typename Executor, typename BaseComponent>
    struct executor_component :  BaseComponent
    {
    private:
        typedef BaseComponent base_type;
        typedef Executor executor_type;
        typedef typename base_type::this_component_type this_component_type;

    public:
        template <typename ... Arg>
        executor_component(executor_type const& exec, Arg &&... arg)
          : base_type(std::forward<Arg>(arg)...),
            exec_(exec)
        {}

        ///////////////////////////////////////////////////////////////////////
        // wrap given function into a nullary function as expected by the
        // executor
        static void execute(hpx::threads::thread_function_type const& f)
        {
            f(hpx::threads::thread_restart_state::signaled);
        }

        /// This is the default hook implementation for schedule_thread which
        /// forwards to the executor instance associated with this component.
#if defined(HPX_HAVE_THREAD_EXECUTORS_COMPATIBILITY)
        template <typename Executor_ = Executor>
        static typename std::enable_if<
            traits::is_threads_executor<Executor_>::value>::type
        schedule_thread(hpx::naming::address::address_type lva,
            naming::address::component_type /* comptype */,
            hpx::threads::thread_init_data& data,
            hpx::threads::thread_schedule_state /* initial_state */)
        {
            hpx::util::thread_description desc(&executor_component::execute);
#ifdef HPX_HAVE_THREAD_DESCRIPTION
            desc = data.description;
#endif
            hpx::parallel::execution::post(
                hpx::get_lva<executor_component>::call(lva)->exec_,
                hpx::util::annotated_function(
                    &executor_component::execute, desc.get_description()),
                std::move(data.func));
        }
#endif

        template <typename Executor_ = Executor>
#if defined(HPX_HAVE_THREAD_EXECUTORS_COMPATIBILITY)
        static typename std::enable_if<
            !traits::is_threads_executor<Executor_>::value>::type
#else
        static void
#endif
        schedule_thread(hpx::naming::address::address_type lva,
            naming::address::component_type /* comptype */,
            hpx::threads::thread_init_data& data,
            hpx::threads::thread_schedule_state /* initial_state */)
        {
            hpx::util::thread_description desc(&executor_component::execute);
#ifdef HPX_HAVE_THREAD_DESCRIPTION
            desc = data.description;
#endif
            hpx::parallel::execution::async_execute(
                hpx::get_lva<executor_component>::call(lva)->exec_,
                hpx::util::deferred_call(
                    hpx::util::annotated_function(
                        &executor_component::execute, desc.get_description()
                    ),
                    std::move(data.func)));
        }

    protected:
        executor_type exec_;
    };
}}
