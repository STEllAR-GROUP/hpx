//  Copyright (c) 2007-2011 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_UTIL_INTERVAL_TIMER_SEP_27_2011_0434PM)
#define HPX_UTIL_INTERVAL_TIMER_SEP_27_2011_0434PM

#include <hpx/hpx_fwd.hpp>
#include <hpx/util/spinlock.hpp>

#include <boost/function.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>

#include <string>
#include <vector>

namespace hpx { namespace util
{
    ///////////////////////////////////////////////////////////////////////////
    class HPX_EXPORT interval_timer
    {
    public:
        interval_timer();
        interval_timer(boost::function<void()> const& f, std::size_t microsecs,
                std::string const& description);
        ~interval_timer();

        void start();
        void stop();

    protected:
        // schedule a high priority task after a given time interval
        void schedule_thread();

        threads::thread_state_enum 
            evaluate(threads::thread_state_ex_enum statex);

    private:
        typedef util::spinlock mutex_type;

        mutable mutex_type mtx_;
        boost::function<void()> f_;   ///< function to call 
        std::size_t microsecs_;       ///< time interval 
        threads::thread_id_type id_;  ///< id of currently scheduled thread
        std::string description_;     ///< description of this interval timer
    };
}}

#endif
