//  Copyright (c) 2007-2015 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_RUNTIME_THREADS_DETAIL_PERIODIC_MAINTENANCE_JAN_11_2015_0626PM)
#define HPX_RUNTIME_THREADS_DETAIL_PERIODIC_MAINTENANCE_JAN_11_2015_0626PM

#include <hpx/config.hpp>
#include <hpx/config/asio.hpp>
#include <hpx/runtime_fwd.hpp>
#include <hpx/state.hpp>
#include <hpx/util/bind.hpp>
#include <hpx/util/date_time_chrono.hpp>
#include <hpx/util/io_service_pool.hpp>

#include <boost/asio/basic_deadline_timer.hpp>
#include <boost/atomic.hpp>
#include <boost/chrono/system_clocks.hpp>
#include <boost/cstdint.hpp>
#include <boost/ref.hpp>

namespace hpx { namespace threads { namespace detail
{
    ///////////////////////////////////////////////////////////////////////////
    inline bool is_running_state(hpx::state state)
    {
        return state == state_running || state == state_suspended;
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename SchedulingPolicy>
    inline void periodic_maintenance_handler(SchedulingPolicy& scheduler,
        boost::atomic<hpx::state>& global_state, std::false_type)
    {
    }

    template <typename SchedulingPolicy>
    inline void periodic_maintenance_handler(SchedulingPolicy& scheduler,
        boost::atomic<hpx::state>& global_state, std::true_type)
    {
        bool running = is_running_state(global_state.load());
        scheduler.periodic_maintenance(running);

        if (running)
        {
            // create timer firing in correspondence with given time
            typedef boost::asio::basic_deadline_timer<
                boost::chrono::steady_clock
              , util::chrono_traits<boost::chrono::steady_clock>
            > deadline_timer;

            deadline_timer t(
                get_thread_pool("timer-thread")->get_io_service(),
                boost::chrono::milliseconds(1000));

            void (*handler)(SchedulingPolicy&, boost::atomic<hpx::state>&,
                std::true_type) =
                &periodic_maintenance_handler<SchedulingPolicy>;

            t.async_wait(util::bind(handler, boost::ref(scheduler),
                boost::ref(global_state), std::true_type()));
        }
    }

    template <typename SchedulingPolicy>
    inline void start_periodic_maintenance(SchedulingPolicy&,
        boost::atomic<hpx::state>& global_state, std::false_type)
    {
    }

    template <typename SchedulingPolicy>
    inline void start_periodic_maintenance(SchedulingPolicy& scheduler,
        boost::atomic<hpx::state>& global_state, std::true_type)
    {
        scheduler.periodic_maintenance(is_running_state(global_state.load()));

        // create timer firing in correspondence with given time
        typedef boost::asio::basic_deadline_timer<
            boost::chrono::steady_clock
          , util::chrono_traits<boost::chrono::steady_clock>
        > deadline_timer;

        deadline_timer t (
            get_thread_pool("io-thread")->get_io_service(),
            boost::chrono::milliseconds(1000));

        void (*handler)(SchedulingPolicy&, boost::atomic<hpx::state>&,
            std::true_type) =
            &periodic_maintenance_handler<SchedulingPolicy>;

        t.async_wait(util::bind(handler, boost::ref(scheduler),
            boost::ref(global_state), std::true_type()));
    }
}}}

#endif


