////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Katelyn Kufahl
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(HPX_05A1C29B_DB73_463A_8C9D_B8EDC3B69F5E)
#define HPX_05A1C29B_DB73_463A_8C9D_B8EDC3B69F5E

#include <boost/config.hpp>
#include <boost/assert.hpp>
#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics.hpp>
#include <boost/accumulators/statistics/mean.hpp>
#include <boost/accumulators/statistics/moment.hpp>
#include <boost/cstdint.hpp>
#include <boost/atomic.hpp>

#include <hpx/performance_counters/parcels/data_point.hpp>
#include <hpx/util/high_resolution_timer.hpp>
#include <hpx/util/spinlock.hpp>

namespace hpx { namespace performance_counters { namespace parcels
{

class gatherer {
  typedef hpx::util::spinlock mutex_type;
  typedef hpx::util::spinlock::scoped_lock lock;


   public:

    gatherer():
      timer(0),
      byte_count(0),
      gatherer_size(0)
    {}

    boost::int64_t size() const;
    void push_back(data_point const& x);

    boost::int64_t mean_time() const;
    boost::int64_t mean_byte() const;
    boost::int64_t mean_time_per_byte() const;

    boost::int64_t moment_time() const;
    boost::int64_t moment_byte() const;
    boost::int64_t moment_time_per_byte() const;

    boost::int64_t variance_time() const;
    boost::int64_t variance_byte() const;
    boost::int64_t variance_time_per_byte() const;

    boost::int64_t total_bytes() const;
    boost::int64_t total_time() const;    

    private:
    util::high_resolution_timer timer;
    boost::atomic<boost::int64_t> byte_count;
    boost::atomic<boost::int64_t> gatherer_size;

    // Create mutexes for accumulator functions.
    mutable mutex_type acc_mtx;

    // Create accumulator sets.
    boost::accumulators::accumulator_set <
        double,
        boost::accumulators::features< boost::accumulators::tag::mean > >
        mean_time_acc, mean_time_per_byte_acc;

    boost::accumulators::accumulator_set <
        boost::int64_t,
        boost::accumulators::features< boost::accumulators::tag::mean> >
        mean_byte_acc;

    boost::accumulators::accumulator_set <
        double,
        boost::accumulators::features< boost::accumulators::tag::moment<2> > >
        moment_time_acc, moment_time_per_byte_acc;
   
    boost::accumulators::accumulator_set<
        boost::int64_t,
        boost::accumulators::features< boost::accumulators::tag::moment<2> > >
        moment_byte_acc;

    boost::accumulators::accumulator_set<
        double,
        boost::accumulators::features< boost::accumulators::tag::variance> >
        variance_time_acc, variance_time_per_byte_acc;

    boost::accumulators::accumulator_set<
        boost::int64_t,
        boost::accumulators::features< boost::accumulators::tag::variance> >
        variance_byte_acc;
};

inline void gatherer::push_back(data_point const& x)
{
    lock mtx(acc_mtx);
   
    byte_count += boost::atomic<boost::int64_t>(x.bytes);
    ++gatherer_size;
    
    mean_time_acc(x.elapsed());
    moment_time_acc(x.elapsed());
    variance_time_acc(x.elapsed());
    mean_byte_acc(x.bytes);
    moment_byte_acc(x.bytes);
    variance_byte_acc(x.bytes);
    BOOST_ASSERT(x.bytes != 0);
    double time_per_byte = x.elapsed() / double(x.bytes);  
    moment_time_per_byte_acc(time_per_byte);
    variance_time_per_byte_acc(time_per_byte);
}

inline boost::int64_t gatherer::size() const
{
    return gatherer_size.load();
}

inline boost::int64_t gatherer::mean_time() const
{
    lock mtx(acc_mtx);
    return boost::int64_t(boost::accumulators::extract::mean(mean_time_acc));
}   
   
inline boost::int64_t gatherer::moment_time() const
{
    lock mtx(acc_mtx);
    return boost::int64_t(boost::accumulators::extract::moment<2>(moment_time_acc));
}

inline boost::int64_t gatherer::variance_time() const
{
    lock mtx(acc_mtx);
    return boost::int64_t(boost::accumulators::extract::variance(variance_time_acc));
}

inline boost::int64_t gatherer::mean_byte() const
{
    lock mtx(acc_mtx); 
    return boost::int64_t(boost::accumulators::extract::mean(mean_byte_acc));
}

inline boost::int64_t gatherer::moment_byte() const
{
    lock mtx(acc_mtx);
    return boost::int64_t(boost::accumulators::extract::moment<2>(moment_byte_acc));
}

inline boost::int64_t gatherer::variance_byte() const
{
    lock mtx(acc_mtx);
    return boost::int64_t(boost::accumulators::extract::variance(variance_byte_acc));
}

inline boost::int64_t gatherer::total_time() const
{
    return boost::int64_t(timer.elapsed());
}

inline boost::int64_t gatherer::total_bytes() const
{
    return byte_count.load();
}

inline boost::int64_t gatherer::mean_time_per_byte() const
{
   return boost::int64_t(total_time() / boost::int64_t(total_bytes()));
}

inline boost::int64_t gatherer::moment_time_per_byte() const
{
    lock mtx(acc_mtx);
    return boost::int64_t(boost::accumulators::extract::moment<2>(moment_time_per_byte_acc));
}

inline boost::int64_t gatherer::variance_time_per_byte() const
{
    lock mtx(acc_mtx);
    return boost::int64_t(boost::accumulators::extract::variance(variance_time_per_byte_acc));
}
}}}

#endif // HPX_05A1C29B_DB73_463A_8C9D_B8EDC3B69F5E

