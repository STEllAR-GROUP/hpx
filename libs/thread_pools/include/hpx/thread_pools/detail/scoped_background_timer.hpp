//  Copyright (c) 2019 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_THREADS_SCOPED_BACKGROUND_TIMER_JUN_10_2019_0301PM)
#define HPX_THREADS_SCOPED_BACKGROUND_TIMER_JUN_10_2019_0301PM

#include <hpx/config.hpp>
#include <hpx/hardware/timestamp.hpp>

#include <cstdint>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace threads {
#if defined(HPX_HAVE_BACKGROUND_THREAD_COUNTERS) &&                            \
    defined(HPX_HAVE_THREAD_IDLE_RATES)
    ////////////////////////////////////////////////////////////////////////////
    struct background_work_duration_counter
    {
        background_work_duration_counter(std::int64_t& background_exec_time)
          : background_exec_time_(background_exec_time)
        {
        }

        void collect_background_exec_time(std::int64_t timestamp)
        {
            if (background_exec_time_ != -1)
            {
                background_exec_time_ +=
                    util::hardware::timestamp() - timestamp;
            }
        }

        std::int64_t& background_exec_time_;
    };

    struct background_exec_time_wrapper
    {
        background_exec_time_wrapper(
            background_work_duration_counter& background_work_duration)
          : timestamp_(background_work_duration.background_exec_time_ != -1 ?
                    util::hardware::timestamp() :
                    -1)
          , background_work_duration_(background_work_duration)
        {
        }

        ~background_exec_time_wrapper()
        {
            background_work_duration_.collect_background_exec_time(timestamp_);
        }

        std::int64_t timestamp_;
        background_work_duration_counter& background_work_duration_;
    };
#endif
}}    // namespace hpx::threads

#endif
