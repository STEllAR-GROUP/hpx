//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#if !defined(HPX_COMPUTE_DEVICE_CODE)
#include <hpx/hpx.hpp>
#include <hpx/include/actions.hpp>
#include <hpx/include/components.hpp>
#include <hpx/include/threads.hpp>
#include <hpx/include/util.hpp>

#include <atomic>

///////////////////////////////////////////////////////////////////////////////
namespace examples { namespace server
{
    ///////////////////////////////////////////////////////////////////////////
    inline void delay(int c)
    {
#if defined(HPX_CLANG_VERSION) && (HPX_CLANG_VERSION >= 100000)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-volatile"
#endif
        double volatile d = 0.;
        for (int i = 0; i < c; ++i)
            d += 1 / (2. * i + 1);
        (void) d;
#if defined(HPX_CLANG_VERSION) && (HPX_CLANG_VERSION >= 100000)
#pragma clang diagnostic pop
#endif
    }

    ///////////////////////////////////////////////////////////////////////////
    class cancelable_action
      : public hpx::components::component_base<cancelable_action>
    {
    private:
        struct reset_id
        {
            explicit reset_id(cancelable_action& this_)
              : outer_(this_)
            {
                {
                    hpx::thread::id old_value =
                        outer_.id_.exchange(hpx::this_thread::get_id());
                    HPX_ASSERT(old_value == hpx::thread::id());
                    HPX_UNUSED(old_value);
                }
            }
            ~reset_id()
            {
                hpx::thread::id old_value =
                    outer_.id_.exchange(hpx::thread::id());
                HPX_ASSERT(old_value != hpx::thread::id());
                HPX_ASSERT(outer_.id_ == hpx::thread::id());
                HPX_UNUSED(old_value);
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
            // Make sure id_ has been set
            hpx::util::yield_while(
                [this]() { return id_ == hpx::thread::id(); });
            HPX_ASSERT(id_ != hpx::thread::id());
            hpx::thread::interrupt(id_);
        }

        HPX_DEFINE_COMPONENT_ACTION(cancelable_action, do_it, do_it_action);
        HPX_DEFINE_COMPONENT_ACTION(cancelable_action, cancel_it, cancel_it_action);

    private:
        std::atomic<hpx::thread::id> id_;
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
