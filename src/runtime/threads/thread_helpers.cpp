//  Copyright (c) 2007-2012 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/threads/thread.hpp>
#include <hpx/runtime/threads/threadmanager.hpp>
#include <hpx/runtime/threads/thread_helpers.hpp>
#include <hpx/runtime/applier/applier.hpp>
#include <hpx/util/stringstream.hpp>

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
                "hpx::threads::set_thread_state",
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
                "hpx::threads::set_thread_state",
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
                "hpx::threads::set_thread_state",
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
                "hpx::threads::get_thread_state",
                "global applier object is not accessible");
            return thread_state(unknown);
        }

        if (&ec != &throws)
            ec = make_success_code();

        return app->get_thread_manager().get_state(id);
    }

    ///////////////////////////////////////////////////////////////////////////
    std::size_t get_thread_phase(thread_id_type id, error_code& ec)
    {
        hpx::applier::applier* app = hpx::applier::get_applier_ptr();
        if (NULL == app)
        {
            HPX_THROWS_IF(ec, invalid_status,
                "hpx::threads::get_thread_state",
                "global applier object is not accessible");
            return std::size_t(~0);
        }

        if (&ec != &throws)
            ec = make_success_code();

        return app->get_thread_manager().get_phase(id);
    }


    ///////////////////////////////////////////////////////////////////////////
    std::string get_thread_description(thread_id_type id, error_code& ec)
    {
        hpx::applier::applier* app = hpx::applier::get_applier_ptr();
        if (NULL == app)
        {
            HPX_THROWS_IF(ec, invalid_status,
                "hpx::threads::get_thread_description",
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
                "hpx::threads::set_thread_description",
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
                "hpx::threads::get_thread_lco_description",
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
                "hpx::threads::set_thread_lco_description",
                "global applier object is not accessible");
            return;
        }

        if (&ec != &throws)
            ec = make_success_code();

        app->get_thread_manager().set_lco_description(id, desc);
    }

    ///////////////////////////////////////////////////////////////////////////
    naming::id_type get_thread_gid(thread_id_type id, error_code& ec)
    {
        hpx::applier::applier* app = hpx::applier::get_applier_ptr();
        if (NULL == app)
        {
            HPX_THROWS_IF(ec, invalid_status,
                "hpx::threads::get_thread_gid",
                "global applier object is not accessible");
            return naming::invalid_id;
        }

        if (&ec != &throws)
            ec = make_success_code();

        return app->get_thread_manager().get_thread_gid(id);
    }

    /// The function \a suspend will return control to the thread manager
    /// (suspends the current thread). It sets the new state of this thread
    /// to the thread state passed as the parameter.
    ///
    /// If the suspension was aborted, this function will throw a
    /// \a yield_aborted exception.
    thread_state_ex_enum suspend(thread_state_enum state,
        char const* description, error_code& ec)
    {
        // let the thread manager do other things while waiting
        threads::thread_self& self = threads::get_self();
        threads::thread_id_type id = self.get_thread_id();
        threads::set_thread_lco_description(id, description, ec);

        threads::thread_state_ex_enum statex = self.yield(state);

        threads::set_thread_lco_description(id, 0, ec);
        if (statex == threads::wait_abort) {
            hpx::util::osstream strm;
            strm << "thread(" << id << ", "
                  << threads::get_thread_description(id)
                  << ") aborted (yield returned wait_abort)";
            HPX_THROWS_IF(ec, yield_aborted, description,
                hpx::util::osstream_get_string(strm));
        }

        if (&ec != &throws)
            ec = make_success_code();

        return statex;
    }

    thread_state_ex_enum suspend(boost::posix_time::ptime const& at_time,
        char const* description, error_code& ec)
    {
        // schedule a thread waking us up at_time
        threads::thread_self& self = threads::get_self();
        threads::thread_id_type id = self.get_thread_id();
        threads::set_thread_lco_description(id, description, ec);

        threads::set_thread_state(id,
            at_time, threads::pending, threads::wait_signaled,
            threads::thread_priority_critical, ec);
        if (ec) return wait_unknown;

        // let the thread manager do other things while waiting
        threads::thread_state_ex_enum statex = self.yield(threads::suspended);

        threads::set_thread_lco_description(id, 0, ec);
        if (statex == threads::wait_abort) {
            hpx::util::osstream strm;
            strm << "thread(" << id << ", "
                  << threads::get_thread_description(id)
                  << ") aborted (yield returned wait_abort)";
            HPX_THROWS_IF(ec, yield_aborted, description,
                hpx::util::osstream_get_string(strm));
        }

        if (&ec != &throws)
            ec = make_success_code();

        return statex;
    }

    thread_state_ex_enum suspend(
        boost::posix_time::time_duration const& after_duration,
        char const* description, error_code& ec)
    {
        // schedule a thread waking us up after_duration
        threads::thread_self& self = threads::get_self();
        threads::thread_id_type id = self.get_thread_id();
        threads::set_thread_lco_description(id, description, ec);

        threads::set_thread_state(id,
            after_duration, threads::pending, threads::wait_signaled,
            threads::thread_priority_critical, ec);
        if (ec) return wait_unknown;

        // let the thread manager do other things while waiting
        threads::thread_state_ex_enum statex = self.yield(threads::suspended);

        threads::set_thread_lco_description(id, 0, ec);
        if (statex == threads::wait_abort) {
            hpx::util::osstream strm;
            strm << "thread(" << id << ", "
                  << threads::get_thread_description(id)
                  << ") aborted (yield returned wait_abort)";
            HPX_THROWS_IF(ec, yield_aborted, description,
                hpx::util::osstream_get_string(strm));
        }

        if (&ec != &throws)
            ec = make_success_code();

        return statex;
    }
}}

