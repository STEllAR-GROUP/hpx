//  Copyright (c) 2007-2008 Anshul Tandon
//  Copyright (c) 2007-2012 Hartmut Kaiser
//  Copyright (c) 2011      Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_fwd.hpp>
#include <hpx/exception.hpp>
#include <hpx/runtime/threads/thread_helpers.hpp>
#include <hpx/runtime/applier/applier.hpp>
#include <hpx/include/parcelset.hpp>
#include <hpx/runtime/agas/interface.hpp>
#include <hpx/runtime/threads/threadmanager.hpp>
#include <hpx/runtime/components/server/runtime_support.hpp>
#include <hpx/util/register_locks.hpp>
#include <hpx/include/async.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace applier
{
    //
    lcos::future<naming::id_type, naming::gid_type>
    create_async(naming::id_type const& targetgid,
        components::component_type type, std::size_t count)
    {
        // Create a future, execute the required action,
        // we simply return the initialized future, the caller needs
        // to call get() on the return value to obtain the result
        typedef
            components::server::runtime_support::create_component_action
        action_type;
        return hpx::async<action_type>(targetgid, type, count);
    }

    //
    naming::id_type create(naming::id_type const& targetgid,
        components::component_type type, std::size_t count)
    {
        return create_async(targetgid, type, count).get();
    }

    ///////////////////////////////////////////////////////////////////////////
    static inline threads::thread_state_enum thread_function(
        HPX_STD_FUNCTION<void(threads::thread_state_ex_enum)> const& func)
    {
        // execute the actual thread function
        func(threads::wait_signaled);

        // Verify that there are no more registered locks for this
        // OS-thread. This will throw if there are still any locks
        // held.
        util::force_error_on_lock();

        return threads::terminated;
    }

    static inline threads::thread_state_enum thread_function_nullary(
        HPX_STD_FUNCTION<void()> const& func)
    {
        // execute the actual thread function
        func();

        // Verify that there are no more registered locks for this
        // OS-thread. This will throw if there are still any locks
        // held.
        util::force_error_on_lock();

        return threads::terminated;
    }

    ///////////////////////////////////////////////////////////////////////////
    threads::thread_id_type register_thread_nullary(
        BOOST_RV_REF(HPX_STD_FUNCTION<void()>) func, char const* desc,
        threads::thread_state_enum state, bool run_now,
        threads::thread_priority priority, std::size_t os_thread, 
        threads::thread_stacksize stacksize, error_code& ec)
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
            HPX_STD_BIND(&thread_function_nullary, boost::move(func)),
            desc ? desc : "<unknown>", 0, priority, os_thread, 
            threads::get_stack_size(stacksize));
        return app->get_thread_manager().
            register_thread(data, state, run_now, ec);
    }

    threads::thread_id_type register_thread(
        BOOST_RV_REF(HPX_STD_FUNCTION<void(threads::thread_state_ex_enum)>) func,
        char const* desc, threads::thread_state_enum state, bool run_now,
        threads::thread_priority priority, std::size_t os_thread, 
        threads::thread_stacksize stacksize, error_code& ec)
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
            HPX_STD_BIND(&thread_function, boost::move(func)),
            desc ? desc : "<unknown>", 0, priority, os_thread,
            threads::get_stack_size(stacksize));
        return app->get_thread_manager().
            register_thread(data, state, run_now, ec);
    }

    threads::thread_id_type register_thread_plain(
        BOOST_RV_REF(HPX_STD_FUNCTION<threads::thread_function_type>) func,
        char const* desc, threads::thread_state_enum state, bool run_now,
        threads::thread_priority priority, std::size_t os_thread, 
        threads::thread_stacksize stacksize, error_code& ec)
    {
        hpx::applier::applier* app = hpx::applier::get_applier_ptr();
        if (NULL == app)
        {
            HPX_THROWS_IF(ec, invalid_status,
                "hpx::applier::register_thread_plain",
                "global applier object is not accessible");
            return threads::invalid_thread_id;
        }

        threads::thread_init_data data(
            boost::move(func), desc ? desc : "<unknown>", 0, priority, 
            os_thread, threads::get_stack_size(stacksize));
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
        BOOST_RV_REF(HPX_STD_FUNCTION<void()>) func, char const* desc,
        threads::thread_state_enum state, threads::thread_priority priority,
        std::size_t os_thread, threads::thread_stacksize stacksize,
        error_code& ec)
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
            HPX_STD_BIND(&thread_function_nullary, boost::move(func)),
            desc ? desc : "<unknown>", 0, priority, os_thread,
            threads::get_stack_size(stacksize));
        app->get_thread_manager().register_work(data, state, ec);
    }

    void register_work(
        BOOST_RV_REF(HPX_STD_FUNCTION<void(threads::thread_state_ex_enum)>) func,
        char const* desc, threads::thread_state_enum state,
        threads::thread_priority priority, std::size_t os_thread, 
        threads::thread_stacksize stacksize, error_code& ec)
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
            HPX_STD_BIND(&thread_function, boost::move(func)),
            desc ? desc : "<unknown>", 0, priority, os_thread,
            threads::get_stack_size(stacksize));
        app->get_thread_manager().register_work(data, state, ec);
    }

    void register_work_plain(
        BOOST_RV_REF(HPX_STD_FUNCTION<threads::thread_function_type>) func,
        char const* desc, naming::address::address_type lva,
        threads::thread_state_enum state, threads::thread_priority priority,
        std::size_t os_thread, threads::thread_stacksize stacksize,
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

        threads::thread_init_data data(boost::move(func),
            desc ? desc : "<unknown>", lva, priority, os_thread,
            threads::get_stack_size(stacksize));
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
    hpx::util::thread_specific_ptr<applier*, applier::tls_tag> applier::applier_;

    applier::applier(parcelset::parcelhandler &ph, threads::threadmanager_base& tm,
                boost::uint64_t rts, boost::uint64_t mem)
      : parcel_handler_(ph), thread_manager_(tm),
        runtime_support_id_(parcel_handler_.get_locality().get_msb(),
            rts, naming::id_type::unmanaged),
        memory_id_(parcel_handler_.get_locality().get_msb(),
            mem, naming::id_type::unmanaged)
    {}

    naming::resolver_client& applier::get_agas_client()
    {
        return hpx::naming::get_agas_client();
    }

    parcelset::parcelhandler& applier::get_parcel_handler()
    {
        return parcel_handler_;
    }

    threads::threadmanager_base& applier::get_thread_manager()
    {
        return thread_manager_;
    }

    naming::locality const& applier::here() const
    {
        return hpx::get_locality();
    }

    naming::gid_type const& applier::get_raw_locality() const
    {
        return hpx::naming::get_agas_client().local_locality();
    }

    boost::uint32_t applier::get_locality_id() const
    {
        return naming::get_locality_id_from_gid(get_raw_locality());
    }

    bool applier::get_raw_remote_localities(std::vector<naming::gid_type>& prefixes,
        components::component_type type) const
    {
        return parcel_handler_.get_raw_remote_localities(prefixes, type);
    }

    bool applier::get_remote_localities(std::vector<naming::id_type>& prefixes,
        components::component_type type) const
    {
        std::vector<naming::gid_type> raw_prefixes;
        if (!parcel_handler_.get_raw_remote_localities(raw_prefixes, type))
            return false;

        BOOST_FOREACH(naming::gid_type& gid, raw_prefixes)
            prefixes.push_back(naming::id_type(gid, naming::id_type::unmanaged));

        return true;
    }

    bool applier::get_raw_localities(std::vector<naming::gid_type>& prefixes,
        components::component_type type) const
    {
        return parcel_handler_.get_raw_localities(prefixes, type);
    }

    bool applier::get_localities(std::vector<naming::id_type>& prefixes,
        components::component_type type) const
    {
        std::vector<naming::gid_type> raw_prefixes;
        if (!parcel_handler_.get_raw_localities(raw_prefixes, type))
            return false;

        BOOST_FOREACH(naming::gid_type& gid, raw_prefixes)
            prefixes.push_back(naming::id_type(gid, naming::id_type::unmanaged));

        return true;
    }

    // parcel forwarding.
    bool applier::route(parcelset::parcel const& p)
    {
        return get_agas_client().route_parcel(p);
    }

    void applier::init_tss()
    {
        if (NULL == applier::applier_.get())
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

    // The function \a get_locality_id returns the id of this locality
    boost::uint32_t get_locality_id()
    {
        applier** appl = applier::applier_.get();
        return appl ? (*appl)->get_locality_id() : naming::invalid_locality_id;
    }
}}

