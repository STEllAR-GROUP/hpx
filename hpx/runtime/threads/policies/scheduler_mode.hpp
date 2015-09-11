//  Copyright (c) 2015 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_THREADS_SCHEDULER_MODE_AUG_27_2017_1136AM)
#define HPX_THREADS_SCHEDULER_MODE_AUG_27_2017_1136AM

namespace hpx { namespace threads { namespace policies
{
    enum scheduler_mode
    {
        nothing_special = 0,
        do_background_work = 0x1,
        reduce_thread_priority = 0x02,
        delay_exit = 0x04,
        fast_idle_mode = 0x08
    };
}}}

#endif

