//  Copyright (c) 2007-2008 Anshul Tandon
//  Copyright (c) 2007-2008 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/include/applier.hpp>
#include <hpx/runtime/actions/continuation_impl.hpp>
#include <hpx/runtime/components/server/runtime_support.hpp>
#include <hpx/lcos/eager_future.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace applier
{
    // 
    lcos::future_value<naming::id_type> 
    create_async(naming::id_type const& targetgid, 
        components::component_type type, std::size_t count)
    {
        // Create a future_value, execute the required action, 
        // we simply return the initialized future_value, the caller needs
        // to call get() on the return value to obtain the result
        typedef 
            components::server::runtime_support::create_component_action
        action_type;
        return lcos::eager_future<action_type>(targetgid, type, count);
    }

    // 
    naming::id_type create(naming::id_type const& targetgid, 
        components::component_type type, std::size_t count)
    {
        return create_async(targetgid, type, count).get();
    }

    //
    void destroy (components::component_type type, naming::id_type const& gid)
    {
        typedef 
            components::server::runtime_support::free_component_action 
        action_type;
        hpx::applier::apply<action_type>(
            hpx::applier::get_applier().get_runtime_support_gid(), type, gid);
    }

    ///////////////////////////////////////////////////////////////////////////
    static threads::thread_state thread_function(boost::function<void()> func)
    {
        func();
        return threads::terminated;
    }

    threads::thread_id_type register_thread(boost::function<void()> func, 
        char const* desc, threads::thread_state state, bool run_now)
    {
        return hpx::applier::get_applier().get_thread_manager().register_thread(
            boost::bind(&thread_function, func), desc, state, run_now);
    }

    threads::thread_id_type register_thread_plain(
        boost::function<threads::thread_function_type> func,
        char const* desc, threads::thread_state state, bool run_now)
    {
        return hpx::applier::get_applier().get_thread_manager().register_thread(
            func, desc, state, run_now);
    }

    ///////////////////////////////////////////////////////////////////////////
    void register_work(boost::function<void()> func, char const* desc, 
        threads::thread_state state, bool run_now)
    {
        hpx::applier::get_applier().get_thread_manager().register_work(
            boost::bind(&thread_function, func), desc, state, run_now);
    }

    void register_work_plain(
        boost::function<threads::thread_function_type> func,
        char const* desc, threads::thread_state state, bool run_now)
    {
        hpx::applier::get_applier().get_thread_manager().register_work(
            func, desc, state, run_now);
    }

    ///////////////////////////////////////////////////////////////////////////
    boost::thread_specific_ptr<applier*> applier::applier_;

    void applier::init_tss()
    {
        BOOST_ASSERT(NULL == applier::applier_.get());    // shouldn't be initialized yet
        applier::applier_.reset(new applier* (this));
    }

    void applier::deinit_tss()
    {
        applier::applier_.reset();
    }

    applier& get_applier()
    {
        BOOST_ASSERT(NULL != applier::applier_.get());   // should have been initialized
        return **applier::applier_;
    }

    applier* get_applier_ptr()
    {
        applier** appl = applier::applier_.get();
        return appl ? *appl : NULL;
    }

///////////////////////////////////////////////////////////////////////////////
}}

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace threads
{
    ///////////////////////////////////////////////////////////////////////////
    thread_state set_thread_state(thread_id_type id, thread_state new_state)
    {
        return hpx::applier::get_applier().get_thread_manager().set_state(id, new_state);
    }

    ///////////////////////////////////////////////////////////////////////////
    thread_id_type set_thread_state(thread_id_type id, 
        boost::posix_time::ptime const& at_time, thread_state state)
    {
        return hpx::applier::get_applier().get_thread_manager().set_state(at_time, id, state);
    }

    ///////////////////////////////////////////////////////////////////////
    thread_id_type set_thread_state(thread_id_type id, 
        boost::posix_time::time_duration const& after, thread_state state)
    {
        return hpx::applier::get_applier().get_thread_manager().set_state(after, id, state);
    }

}}
