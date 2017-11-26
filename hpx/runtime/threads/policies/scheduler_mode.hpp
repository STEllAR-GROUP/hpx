//  Copyright (c) 2015-2017 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_THREADS_SCHEDULER_MODE_AUG_27_2017_1136AM)
#define HPX_THREADS_SCHEDULER_MODE_AUG_27_2017_1136AM

namespace hpx { namespace threads { namespace policies
{
    enum scheduler_mode
    {
        nothing_special = 0,            ///< As the name suggests, this option
            ///< can be used to disable all other options.
        do_background_work = 0x1,       ///< The scheduler will periodically
            ///< call a provided callback function from a special HPX thread
            ///< to enable performing background-work, for instance driving
            ///< networking progress or garbage-collect AGAS
        reduce_thread_priority = 0x02,  ///< The kernel priority of the
            ///< os-thread driving the scheduler will be reduced below normal.
        delay_exit = 0x04,              ///< The scheduler will wait for some
            ///< unspecified amount of time before exiting the scheduling loop
            ///< while being terminated to make sure no other work is being
            ///< scheduled during processing the shutdown request.
        fast_idle_mode = 0x08,          ///< Some schedulers have the capability
            ///< to act as 'embedded' schedulers. In this case it needs to
            ///< periodically invoke a provided callback into the outer scheduler
            ///< more frequently than normal. This option enables this behavior.
        enable_elasticity = 0x10        ///< This options allows for the
            ///< scheduler to dynamically increase and reduce the number of
            ///< processing units it runs on.
    };
}}}

#endif

