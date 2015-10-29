//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_THROTTLE_SERVER_AUG_09_2011_0702PM)
#define HPX_THROTTLE_SERVER_AUG_09_2011_0702PM

#include <hpx/hpx_fwd.hpp>
#include <hpx/include/components.hpp>
#include <hpx/lcos/future.hpp>
#include <hpx/lcos/local/mutex.hpp>
#include <hpx/runtime/actions/component_action.hpp>

#include <boost/dynamic_bitset.hpp>

namespace throttle { namespace server
{
    ///////////////////////////////////////////////////////////////////////////
    class HPX_COMPONENT_EXPORT throttle
      : public hpx::components::component_base<throttle>
    {
    private:
        typedef hpx::components::component_base<throttle> base_type;
        typedef hpx::lcos::local::mutex mutex_type;

    public:
        throttle();
        ~throttle();

        // components must contain a typedef for wrapping_type defining the
        // component type used to encapsulate instances of this component
        typedef throttle wrapping_type;

        ///////////////////////////////////////////////////////////////////////
        // parcel action code: the action to be performed on the destination
        // object (the accumulator)
        enum actions
        {
            throttle_suspend = 0,
            throttle_resume = 1,
            throttle_is_suspended = 2
        };

        void suspend(std::size_t shepherd);
        void resume(std::size_t shepherd);
        bool is_suspended(std::size_t shepherd) const;

        ///////////////////////////////////////////////////////////////////////
        // Each of the exposed functions needs to be encapsulated into an action
        // type, allowing to generate all required boilerplate code for threads,
        // serialization, etc.
        HPX_DEFINE_COMPONENT_ACTION(throttle, suspend, suspend_action);
        HPX_DEFINE_COMPONENT_ACTION(throttle, resume, resume_action);
        HPX_DEFINE_COMPONENT_ACTION(throttle, is_suspended, is_suspended_action);

    private:
        // this function is periodically scheduled as a worker thread with the
        // aim of blocking the execution of its shepherd thread
        void throttle_controller(std::size_t shepherd);

        // schedule a high priority task on the given shepherd thread
        void register_thread(std::size_t shepherd);
        void register_suspend_thread(std::size_t shepherd);

        // this is a bit mask where any set bit means the corresponding
        // shepherd is to be blocked
        boost::dynamic_bitset<> blocked_os_threads_;
        mutable mutex_type mtx_;
    };
}}

HPX_ACTION_HAS_CRITICAL_PRIORITY(throttle::server::throttle::suspend_action);
HPX_REGISTER_ACTION_DECLARATION(
    throttle::server::throttle::suspend_action
  , throttle_suspend_action);

HPX_ACTION_HAS_CRITICAL_PRIORITY(throttle::server::throttle::resume_action);
HPX_REGISTER_ACTION_DECLARATION(
    throttle::server::throttle::resume_action
  , throttle_resume_action);

HPX_ACTION_HAS_CRITICAL_PRIORITY(throttle::server::throttle::is_suspended_action);
HPX_REGISTER_ACTION_DECLARATION(
    throttle::server::throttle::is_suspended_action
  , throttle_is_suspended_action);

#endif
