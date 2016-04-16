//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_EXAMPLE_CANCELABLE_ACTION_APR_21_2012_0955AM)
#define HPX_EXAMPLE_CANCELABLE_ACTION_APR_21_2012_0955AM

#include <hpx/hpx_fwd.hpp>
#include <hpx/include/components.hpp>
#include <hpx/include/actions.hpp>
#include <hpx/include/threads.hpp>

#include <mutex>

///////////////////////////////////////////////////////////////////////////////
namespace examples { namespace server
{
    ///////////////////////////////////////////////////////////////////////////
    inline void delay(int c)
    {
        double volatile d = 0.;
        for (int i = 0; i < c; ++i)
            d += 1 / (2. * i + 1);
    }

    ///////////////////////////////////////////////////////////////////////////
    class cancelable_action
      : public hpx::components::component_base<cancelable_action>
    {
    private:
        typedef hpx::lcos::local::spinlock mutex_type;

        struct reset_id
        {
            reset_id(cancelable_action& this_)
              : outer_(this_)
            {
                std::lock_guard<mutex_type> l(outer_.mtx_);
                outer_.id_ = hpx::this_thread::get_id();
            }
            ~reset_id()
            {
                std::lock_guard<mutex_type> l(outer_.mtx_);
                outer_.id_ = hpx::thread::id();    // invalidate thread id
            }

            cancelable_action& outer_;
        };

    public:
        // Do some lengthy work
        void do_it()
        {
            reset_id r(*this);      // manage thread id

            while (true) {
                // do something useful ;-)
                delay(1000);

                // check whether this thread was interrupted and throw a special
                // exception if it was interrupted
                hpx::this_thread::suspend(); // interruption_point();
            }
        }

        // Cancel the lengthy action above
        void cancel_it()
        {
            std::lock_guard<mutex_type> l(mtx_);
            if (id_ != hpx::thread::id()) {
                hpx::thread::interrupt(id_);
                id_ = hpx::thread::id();        // invalidate thread id
            }
        }

        HPX_DEFINE_COMPONENT_ACTION(cancelable_action, do_it, do_it_action);
        HPX_DEFINE_COMPONENT_ACTION(cancelable_action, cancel_it, cancel_it_action);

    private:
        mutable mutex_type mtx_;
        hpx::thread::id id_;
    };
}}

///////////////////////////////////////////////////////////////////////////////
HPX_REGISTER_ACTION_DECLARATION(
    examples::server::cancelable_action::do_it_action,
    cancelable_action_do_it_action);
HPX_REGISTER_ACTION_DECLARATION(
    examples::server::cancelable_action::cancel_it_action,
    cancelable_action_cancel_it_action);

#endif

