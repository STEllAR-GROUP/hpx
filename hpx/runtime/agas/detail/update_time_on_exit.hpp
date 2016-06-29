//  Copyright (c) 2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_AGAS_UPDATE_TIME_ON_EXIT_JUN_29_2106_0158PM)
#define HPX_AGAS_UPDATE_TIME_ON_EXIT_JUN_29_2106_0158PM

#include <hpx/config.hpp>
#include <hpx/util/high_resolution_clock.hpp>

#include <boost/atomic.hpp>
#include <boost/cstdint.hpp>

namespace hpx { namespace agas { namespace detail
{
    struct update_time_on_exit
    {
        update_time_on_exit(boost::atomic<boost::int64_t>& t)
          : started_at_(hpx::util::high_resolution_clock::now())
          , t_(t)
        {}

        ~update_time_on_exit()
        {
            t_ += (hpx::util::high_resolution_clock::now() - started_at_);
        }

        boost::uint64_t started_at_;
        boost::atomic<boost::int64_t>& t_;
    };
}}}

#endif

