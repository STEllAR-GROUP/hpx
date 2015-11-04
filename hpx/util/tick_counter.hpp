//  Copyright (c) 2005-2012 Hartmut Kaiser
//  Copyright (c) 2014      Bryce Adelstein-Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_UTIL_TICK_COUNTER_MAR_24_2008_1222PM)
#define HPX_UTIL_TICK_COUNTER_MAR_24_2008_1222PM

#include <hpx/util/hardware/timestamp.hpp>

#include <boost/cstdint.hpp>

namespace hpx { namespace util
{
    ///////////////////////////////////////////////////////////////////////////
    //
    //  tick_counter - a timer
    //
    ///////////////////////////////////////////////////////////////////////////
    class tick_counter
    {
    public:
        tick_counter(boost::uint64_t& output)
          : start_time_(take_time_stamp())
          , output_(output)
        {}

        ~tick_counter()
        {
            output_ += take_time_stamp() - start_time_;
        }

    protected:
        static boost::uint64_t take_time_stamp()
        {
            return hpx::util::hardware::timestamp();
        }

    private:
        boost::uint64_t const start_time_;
        boost::uint64_t& output_;
    };
}} // namespace hpx::util

#endif

