//  (C) Copyright 2005-7 Anthony Williams
//  (C) Copyright 2005 John Maddock
//  (C) Copyright 2011-2012 Vicente J. Botet Escriba
//  Copyright (c) 2013 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See
//  accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_LCOS_LOCAL_ONCE_JAN_03_2013_0810PM)
#define HPX_LCOS_LOCAL_ONCE_JAN_03_2013_0810PM

#include <hpx/hpx_fwd.hpp>
#include <hpx/config/emulate_deleted.hpp>
#include <hpx/lcos/local/event.hpp>
#include <hpx/lcos/local/once_fwd.hpp>
#include <hpx/util/assert.hpp>

#include <boost/atomic.hpp>

namespace hpx { namespace lcos { namespace local
{
    struct once_flag
    {
        once_flag() HPX_NOEXCEPT
          : status_(0)
        {}

        HPX_NON_COPYABLE(once_flag)

    private:
        boost::atomic<long> status_;
        lcos::local::event event_;

        template <typename Function>
        friend void call_once(once_flag& flag, Function f);
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename Function>
    void call_once(once_flag& flag, Function f)
    {
        // Try for a quick win: if the procedure has already been called
        // just skip through:
        long const function_complete_flag_value = 0xc15730e2;
        long const running_value = 0x7f0725e3;

        while (flag.status_.load(boost::memory_order_acquire) !=
            function_complete_flag_value)
        {
            long status = 0;
            if (flag.status_.compare_exchange_strong(status, running_value))
            {
                try {
                    // reset event to ensure its usability in case the
                    // wrapped function was throwing an exception before
                    flag.event_.reset();

                    f();

                    // set status to done, release waiting threads
                    flag.status_.store(function_complete_flag_value);
                    flag.event_.set();
                    break;
                }
                catch (...) {
                    // reset status to initial, release waiting threads
                    flag.status_.store(0);
                    flag.event_.set();

                    throw;
                }
            }

            // we're done if function was called
            if (status == function_complete_flag_value)
                break;

            // wait for the function finish executing
            flag.event_.wait();
        }
    }
}}}

#endif
