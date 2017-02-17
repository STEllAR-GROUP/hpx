//  Copyright (c) 2017 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_RUNTIME_COMPONENTS_SERVER_EXECUTOR_COMPONENT_FEB_09_2017_0839PM)
#define HPX_RUNTIME_COMPONENTS_SERVER_EXECUTOR_COMPONENT_FEB_09_2017_0839PM

#include <hpx/config.hpp>
#include <hpx/parallel/executors/executor_traits.hpp>
#include <hpx/runtime/get_lva.hpp>
#include <hpx/runtime/naming/address.hpp>
#include <hpx/runtime/threads/executors.hpp>
#include <hpx/runtime/threads/thread_data_fwd.hpp>
#include <hpx/runtime/threads/thread_enums.hpp>
#include <hpx/traits/is_launch_policy.hpp>
#include <hpx/util/annotated_function.hpp>
#include <hpx/util/deferred_call.hpp>
#include <hpx/util/thread_description.hpp>

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
            f(hpx::threads::wait_signaled);
        }

        /// This is the default hook implementation for schedule_thread which
        /// forwards to the executor instance associated with this component.
        template <typename Executor_ = Executor>
        static typename std::enable_if<
            traits::is_threads_executor<Executor_>::value
        >::type
        schedule_thread(hpx::naming::address::address_type lva,
            hpx::threads::thread_init_data& data,
            hpx::threads::thread_state_enum initial_state)
        {
            hpx::util::thread_description desc(&executor_component::execute);
#ifdef HPX_HAVE_THREAD_DESCRIPTION
            desc = data.description;
#endif
            hpx::get_lva<executor_component>::call(lva)->exec_.add(
                hpx::util::deferred_call(&executor_component::execute,
                    std::move(data.func)),
                desc, initial_state);
        }

        template <typename Executor_ = Executor>
        static typename std::enable_if<
            !traits::is_threads_executor<Executor_>::value
        >::type
        schedule_thread(hpx::naming::address::address_type lva,
            hpx::threads::thread_init_data& data,
            hpx::threads::thread_state_enum initial_state)
        {
            hpx::util::thread_description desc(&executor_component::execute);
#ifdef HPX_HAVE_THREAD_DESCRIPTION
            desc = data.description;
#endif
            hpx::parallel::executor_traits<executor_type>::async_execute(
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

#endif

