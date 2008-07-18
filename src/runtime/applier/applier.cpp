//  Copyright (c) 2007-2008 Anshul Tandon
//  Copyright (c) 2007-2008 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/include/applier.hpp>
#include <hpx/components/server/managed_component_base.hpp>
#include <hpx/runtime/actions/continuation_impl.hpp>
#include <hpx/lcos/eager_future.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace applier
{
    // 
    lcos::simple_future<naming::id_type> 
    create_async(applier& appl, naming::id_type const& targetgid, 
        components::component_type type, std::size_t count)
    {
        // Create a simple_future, execute the required action, 
        // we simply return the initialized simple_future, the caller needs
        // to call get_result() on the return value to obtain the result
        typedef 
            components::server::runtime_support::create_component_action
        action_type;
        return lcos::eager_future<action_type, naming::id_type>(appl, 
            targetgid, type, count);
    }

    // 
    naming::id_type create(applier& appl, threads::thread_self& self,
        naming::id_type const& targetgid, components::component_type type,
        std::size_t count)
    {
        return create_async(appl, targetgid, type, count).get_result(self);
    }

    //
    void destroy (applier& appl, components::component_type type, 
        naming::id_type const& gid, std::size_t count)
    {
        typedef 
            components::server::runtime_support::free_component_action 
        action_type;
        appl.apply<action_type>(appl.get_runtime_support_gid(), type, gid, count);
    }

    threads::thread_id_type register_work(applier& appl,
        boost::function<threads::thread_function_type> func,
        threads::thread_state state, bool run_now)
    {
        return appl.get_thread_manager().register_work(func, state, run_now);
    }

    threads::thread_id_type register_work(applier& appl,
        full_thread_function_type* func, threads::thread_state state, 
        bool run_now)
    {
        return appl.get_thread_manager().register_work(
            boost::bind(func, _1, boost::ref(appl)), state, run_now);
    }

///////////////////////////////////////////////////////////////////////////////
}}

