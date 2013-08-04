//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/actions/continuation.hpp>
#include <hpx/runtime/components/component_factory.hpp>
#include <hpx/runtime.hpp>
#include <hpx/util/scoped_unlock.hpp>

#include <hpx/util/portable_binary_iarchive.hpp>
#include <hpx/util/portable_binary_oarchive.hpp>

#include <boost/assert.hpp>
#include <boost/bind.hpp>
#include <boost/thread.hpp>

#include "throttle.hpp"

///////////////////////////////////////////////////////////////////////////////
namespace throttle { namespace server
{
    throttle::throttle()
    {
        const std::size_t num_threads = hpx::get_os_thread_count();
        BOOST_ASSERT(num_threads != std::size_t(-1));
        blocked_os_threads_.resize(num_threads);

        std::cerr << "Created throttle component!" << std::endl;
    }

    throttle::~throttle()
    {
        std::cerr << "Released throttle component!" << std::endl;
    }

    void throttle::suspend(std::size_t shepherd)
    {
        // If the current thread is not the requested one, re-schedule a new
        // PX thread in order to retry.
        std::size_t thread_num = hpx::get_worker_thread_num();
        if (thread_num != shepherd) {
            register_suspend_thread(shepherd);
            return;
        }

        mutex_type::scoped_lock l(mtx_);

        if (shepherd >= blocked_os_threads_.size()) {
            HPX_THROW_EXCEPTION(hpx::bad_parameter, "throttle::suspend",
                "invalid thread number");
        }

        bool is_suspended = blocked_os_threads_[shepherd];
        if (!is_suspended) {
            blocked_os_threads_[shepherd] = true;
            register_thread(shepherd);
        }
    }

    void throttle::resume(std::size_t shepherd)
    {
        mutex_type::scoped_lock l(mtx_);

        if (shepherd >= blocked_os_threads_.size()) {
            HPX_THROW_EXCEPTION(hpx::bad_parameter, "throttle::resume",
                "invalid thread number");
        }

        blocked_os_threads_[shepherd] = false;   // re-activate shepherd
    }

    bool throttle::is_suspended(std::size_t shepherd) const
    {
        mutex_type::scoped_lock l(mtx_);

        if (shepherd >= blocked_os_threads_.size()) {
            HPX_THROW_EXCEPTION(hpx::bad_parameter, "throttle::is_suspended",
                "invalid thread number");
        }

        return blocked_os_threads_[shepherd];
    }

    // do the requested throttling
    void throttle::throttle_controller(std::size_t shepherd)
    {
        mutex_type::scoped_lock l(mtx_);
        if (!blocked_os_threads_[shepherd])
            return;     // nothing more to do

        {
            // put this shepherd thread to sleep for 100ms
            boost::system_time xt(boost::get_system_time() +
                boost::posix_time::milliseconds(100));

            hpx::util::scoped_unlock<mutex_type::scoped_lock> ul(l);
            boost::thread::sleep(xt);
        }

        // if this thread still needs to be suspended, re-schedule this routine
        // which will give the thread manager some cycles to tend to the high
        // priority tasks which might have arrived
        if (blocked_os_threads_[shepherd])
            register_thread(shepherd);
    }

    // schedule a high priority task on the given shepherd thread
    void throttle::register_thread(std::size_t shepherd)
    {
        std::string description("throttle controller for shepherd thread (" +
            boost::lexical_cast<std::string>(shepherd) + ")");

        hpx::applier::register_thread(
            boost::bind(&throttle::throttle_controller, this, shepherd),
            description.c_str(),
            hpx::threads::pending, true,
            hpx::threads::thread_priority_critical,
            shepherd);
    }

    // schedule a high priority task on the given shepherd thread to suspend
    void throttle::register_suspend_thread(std::size_t shepherd)
    {
        std::string description("suspend shepherd thread (" +
            boost::lexical_cast<std::string>(shepherd) + ")");

        hpx::applier::register_thread(
            boost::bind(&throttle::suspend, this, shepherd),
            description.c_str(),
            hpx::threads::pending, true,
            hpx::threads::thread_priority_critical,
            shepherd);
    }
}}

///////////////////////////////////////////////////////////////////////////////
typedef throttle::server::throttle throttle_type;

HPX_REGISTER_MINIMAL_COMPONENT_FACTORY(
    hpx::components::simple_component<throttle_type>, throttle_throttle_type);

///////////////////////////////////////////////////////////////////////////////
// Serialization support for the actions
HPX_REGISTER_ACTION(throttle_type::suspend_action, throttle_suspend_action);
HPX_REGISTER_ACTION(throttle_type::resume_action, throttle_resume_action);
HPX_REGISTER_ACTION(throttle_type::is_suspended_action, throttle_is_suspended_action);

