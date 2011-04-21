//  Copyright (c) 2007-2008 Anshul Tandon
//  Copyright (c) 2007-2011 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/threads/thread_helpers.hpp>
#include <hpx/runtime/applier/applier.hpp>
#include <hpx/runtime/actions/continuation_impl.hpp>
#include <hpx/runtime/components/server/runtime_support.hpp>
#include <hpx/lcos/eager_future.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace applier
{
    // 
    lcos::future_value<naming::id_type, naming::gid_type> 
    create_async(naming::id_type const& targetgid, 
        components::component_type type, std::size_t count)
    {
        // Create a future_value, execute the required action, 
        // we simply return the initialized future_value, the caller needs
        // to call get() on the return value to obtain the result
        typedef 
            components::server::runtime_support::create_component_action
        action_type;
        return lcos::eager_future<action_type, naming::id_type>(targetgid, type, count);
    }

    // 
    naming::id_type create(naming::id_type const& targetgid, 
        components::component_type type, std::size_t count)
    {
        return create_async(targetgid, type, count).get();
    }

    ///////////////////////////////////////////////////////////////////////////
    static inline threads::thread_state thread_function(
        boost::function<void(threads::thread_state_ex)> const& func)
    {
        func(threads::thread_state_ex(threads::wait_signaled));
        return threads::thread_state(threads::terminated);
    }

    static inline threads::thread_state thread_function_nullary(
        boost::function<void()> const& func)
    {
        func();
        return threads::thread_state(threads::terminated);
    }

    ///////////////////////////////////////////////////////////////////////////
    threads::thread_id_type register_thread_nullary(
        boost::function<void()> const& func, char const* desc, 
        threads::thread_state_enum state, threads::thread_priority priority,
        bool run_now, std::size_t os_thread, error_code& ec)
    {
        hpx::applier::applier* app = hpx::applier::get_applier_ptr();
        if (NULL == app)
        {
            HPX_THROWS_IF(ec, invalid_status, 
                "hpx::applier::register_thread_nullary", 
                "global applier object is not accessible");
            return threads::invalid_thread_id;
        }

        threads::thread_init_data data(
            boost::bind(&thread_function_nullary, func), desc, 0, 
            priority, os_thread);
        return app->get_thread_manager().
            register_thread(data, state, run_now, ec);
    }

    threads::thread_id_type register_thread(
        boost::function<void(threads::thread_state_ex)> const& func, 
        char const* desc, threads::thread_state_enum state, bool run_now,
        threads::thread_priority priority, std::size_t os_thread, error_code& ec)
    {
        hpx::applier::applier* app = hpx::applier::get_applier_ptr();
        if (NULL == app)
        {
            HPX_THROWS_IF(ec, invalid_status, 
                "hpx::applier::register_thread", 
                "global applier object is not accessible");
            return threads::invalid_thread_id;
        }

        threads::thread_init_data data(
            boost::bind(&thread_function, func), desc, 0, priority, os_thread);
        return app->get_thread_manager().
            register_thread(data, state, run_now, ec);
    }

    threads::thread_id_type register_thread_plain(
        boost::function<threads::thread_function_type> const& func,
        char const* desc, threads::thread_state_enum state, bool run_now,
        threads::thread_priority priority, std::size_t os_thread, error_code& ec)
    {
        hpx::applier::applier* app = hpx::applier::get_applier_ptr();
        if (NULL == app)
        {
            HPX_THROWS_IF(ec, invalid_status, 
                "hpx::applier::register_thread_plain", 
                "global applier object is not accessible");
            return threads::invalid_thread_id;
        }

        threads::thread_init_data data(func, desc, 0, priority, os_thread);
        return app->get_thread_manager().
            register_thread(data, state, run_now, ec);
    }

    threads::thread_id_type register_thread_plain(
        threads::thread_init_data& data, threads::thread_state_enum state, 
        bool run_now, error_code& ec)
    {
        hpx::applier::applier* app = hpx::applier::get_applier_ptr();
        if (NULL == app)
        {
            HPX_THROWS_IF(ec, invalid_status, 
                "hpx::applier::register_thread_plain", 
                "global applier object is not accessible");
            return threads::invalid_thread_id;
        }

        return app->get_thread_manager().
            register_thread(data, state, run_now, ec);
    }

    ///////////////////////////////////////////////////////////////////////////
    void register_work_nullary(
        boost::function<void()> const& func, char const* desc, 
        threads::thread_state_enum state, threads::thread_priority priority,
        std::size_t os_thread, error_code& ec)
    {
        hpx::applier::applier* app = hpx::applier::get_applier_ptr();
        if (NULL == app)
        {
            HPX_THROWS_IF(ec, invalid_status, 
                "hpx::applier::register_work_nullary", 
                "global applier object is not accessible");
            return;
        }

        threads::thread_init_data data(
            boost::bind(&thread_function_nullary, func), 
            desc ? desc : "<unknown>", 0, priority, os_thread);
        app->get_thread_manager().register_work(data, state, ec);
    }

    void register_work(
        boost::function<void(threads::thread_state_ex)> const& func, 
        char const* desc, threads::thread_state_enum state, 
        threads::thread_priority priority, std::size_t os_thread, error_code& ec)
    {
        hpx::applier::applier* app = hpx::applier::get_applier_ptr();
        if (NULL == app)
        {
            HPX_THROWS_IF(ec, invalid_status, 
                "hpx::applier::register_work", 
                "global applier object is not accessible");
            return;
        }

        threads::thread_init_data data(
            boost::bind(&thread_function, func), 
            desc ? desc : "<unknown>", 0, priority, os_thread);
        app->get_thread_manager().register_work(data, state, ec);
    }

    void register_work_plain(
        boost::function<threads::thread_function_type> const& func,
        char const* desc, naming::address::address_type lva,
        threads::thread_state_enum state, threads::thread_priority priority,
        std::size_t os_thread, error_code& ec)
    {
        hpx::applier::applier* app = hpx::applier::get_applier_ptr();
        if (NULL == app)
        {
            HPX_THROWS_IF(ec, invalid_status, 
                "hpx::applier::register_work_plain", 
                "global applier object is not accessible");
            return;
        }

        threads::thread_init_data data(func, 
            desc ? desc : "<unknown>", lva, priority, os_thread);
        app->get_thread_manager().register_work(data, state, ec);
    }

    void register_work_plain(
        threads::thread_init_data& data, threads::thread_state_enum state,
        error_code& ec)
    {
        hpx::applier::applier* app = hpx::applier::get_applier_ptr();
        if (NULL == app)
        {
            HPX_THROWS_IF(ec, invalid_status, 
                "hpx::applier::register_work_plain", 
                "global applier object is not accessible");
            return;
        }

        app->get_thread_manager().register_work(data, state, ec);
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

    // The function \a get_prefix_id returns the id of this locality
    boost::uint32_t get_prefix_id()
    {
        applier** appl = applier::applier_.get();
        return appl ? (*appl)->get_prefix_id() : 0;
    }

///////////////////////////////////////////////////////////////////////////////
}}

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace threads
{
    ///////////////////////////////////////////////////////////////////////////
    thread_state set_thread_state(thread_id_type id, thread_state_enum state,
        thread_state_ex_enum stateex, thread_priority priority, error_code& ec)
    {
        hpx::applier::applier* app = hpx::applier::get_applier_ptr();
        if (NULL == app)
        {
            HPX_THROWS_IF(ec, invalid_status, 
                "hpx::applier::set_thread_state", 
                "global applier object is not accessible");
            return thread_state(unknown);
        }

        if (&ec != &throws)
            ec = make_success_code();

        return app->get_thread_manager().set_state(id, state, stateex, 
            priority, ec);
    }

    ///////////////////////////////////////////////////////////////////////////
    thread_id_type set_thread_state(thread_id_type id, 
        boost::posix_time::ptime const& at_time, thread_state_enum state,
        thread_state_ex_enum stateex, thread_priority priority, error_code& ec)
    {
        hpx::applier::applier* app = hpx::applier::get_applier_ptr();
        if (NULL == app)
        {
            HPX_THROWS_IF(ec, invalid_status, 
                "hpx::applier::set_thread_state", 
                "global applier object is not accessible");
            return invalid_thread_id;
        }

        if (&ec != &throws)
            ec = make_success_code();

        return app->get_thread_manager().set_state(at_time, id, state, 
            stateex, priority, ec);
    }

    ///////////////////////////////////////////////////////////////////////////
    thread_id_type set_thread_state(thread_id_type id, 
        boost::posix_time::time_duration const& after, thread_state_enum state,
        thread_state_ex_enum stateex, thread_priority priority, error_code& ec)
    {
        hpx::applier::applier* app = hpx::applier::get_applier_ptr();
        if (NULL == app)
        {
            HPX_THROWS_IF(ec, invalid_status, 
                "hpx::applier::set_thread_state", 
                "global applier object is not accessible");
            return invalid_thread_id;
        }

        if (&ec != &throws)
            ec = make_success_code();

        return app->get_thread_manager().set_state(after, id, state, 
            stateex, priority, ec);
    }

    ///////////////////////////////////////////////////////////////////////////
    thread_state get_thread_state(thread_id_type id, error_code& ec)
    {
        hpx::applier::applier* app = hpx::applier::get_applier_ptr();
        if (NULL == app)
        {
            HPX_THROWS_IF(ec, invalid_status, 
                "hpx::applier::get_thread_state", 
                "global applier object is not accessible");
            return thread_state(unknown);
        }

        if (&ec != &throws)
            ec = make_success_code();

        return app->get_thread_manager().get_state(id);
    }

    ///////////////////////////////////////////////////////////////////////////
    std::string get_thread_description(thread_id_type id, error_code& ec)
    {
        hpx::applier::applier* app = hpx::applier::get_applier_ptr();
        if (NULL == app)
        {
            HPX_THROWS_IF(ec, invalid_status, 
                "hpx::applier::get_thread_description", 
                "global applier object is not accessible");
            return std::string();
        }

        if (&ec != &throws)
            ec = make_success_code();

        return app->get_thread_manager().get_description(id);
    }
    void set_thread_description(thread_id_type id, char const* desc, 
        error_code& ec)
    {
        hpx::applier::applier* app = hpx::applier::get_applier_ptr();
        if (NULL == app)
        {
            HPX_THROWS_IF(ec, invalid_status, 
                "hpx::applier::set_thread_description", 
                "global applier object is not accessible");
            return;
        }

        if (&ec != &throws)
            ec = make_success_code();

        app->get_thread_manager().set_description(id, desc);
    }

    std::string get_thread_lco_description(thread_id_type id, error_code& ec)
    {
        hpx::applier::applier* app = hpx::applier::get_applier_ptr();
        if (NULL == app)
        {
            HPX_THROWS_IF(ec, invalid_status, 
                "hpx::applier::get_thread_lco_description", 
                "global applier object is not accessible");
            return std::string();
        }

        if (&ec != &throws)
            ec = make_success_code();

        return app->get_thread_manager().get_lco_description(id);
    }
    void set_thread_lco_description(thread_id_type id, char const* desc, 
        error_code& ec)
    {
        hpx::applier::applier* app = hpx::applier::get_applier_ptr();
        if (NULL == app)
        {
            HPX_THROWS_IF(ec, invalid_status, 
                "hpx::applier::set_thread_lco_description", 
                "global applier object is not accessible");
            return;
        }

        if (&ec != &throws)
            ec = make_success_code();

        app->get_thread_manager().set_lco_description(id, desc);
    }

    ///////////////////////////////////////////////////////////////////////////
    naming::id_type const& get_thread_gid(thread_id_type id, error_code& ec)
    {
        hpx::applier::applier* app = hpx::applier::get_applier_ptr();
        if (NULL == app)
        {
            HPX_THROWS_IF(ec, invalid_status, 
                "hpx::applier::get_thread_gid", 
                "global applier object is not accessible");
            return naming::invalid_id;
        }

        if (&ec != &throws)
            ec = make_success_code();

        return app->get_thread_manager().get_thread_gid(id);
    }
}}
