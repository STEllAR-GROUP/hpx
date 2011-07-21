////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Katelyn Kufahl
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(HPX_D2C46275_9DC4_4957_B0EF_7B796F0DF254)
#define HPX_D2C46275_9DC4_4957_B0EF_7B796F0DF254

#include <hpx/runtime/naming/name.hpp>
#include <boost/atomic.hpp>
#include <boost/cstdint.hpp>

namespace hpx { namespace performance_counters { namespace parcels
{

class count_and_time_data_point {

    public:

    data_point(boost::atomic<boost::int64_t> count_, boost::atomic<double> time_):

        count(count_)
      , time(time_)
      {}

    data_point():

        count(0)
      , time(0)
      {}

    data_point(data_point const& x):

        count(x.count)
      , time(x.time)
      {}

    boost::atomic<boost::int64_t> count;
    double time;
};

}}}

#endif // HPX_D2C46275_9DC4_4957_B0EF_7B796F0DF254

