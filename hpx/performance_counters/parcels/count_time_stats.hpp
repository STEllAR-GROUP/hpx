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

#include <hpx/performance_counters/parcels/count_and_time_data_point.hpp>
#include <hpx/util/high_resolution_timer.hpp>
#include <hpx/util/spinlock.hpp>

namespace hpx { namespace performance_counters { namespace parcels
{

class count_time_stats {
  typedef hpx::util::spinlock mutex_type;
  typedef mutex_type::scoped_lock lock;
  typedef boost::accumulators::accumulator_set set;

   public:

    count_time_stats():
      count_time_stats_size(0)
    {}

    boost::int64_t size() const;
    void push_back(data_point const& x);

    double mean_time() const;
    double moment_time() const;
    double variance_time() const;

    double total_time() const;    

    private:
    util::high_resolution_timer timer;
    boost::atomic<boost::int64_t> count_time_stats_size;

    // Create mutexes for accumulator functions.
    mutable mutex_type mean_time_mtx;
    mutable mutex_type moment_time_mtx;
    mutable mutex_type variance_time_mtx;

    // Create accumulator sets.
    set < double,
        boost::accumulators::features< boost::accumulators::tag::mean > >
        mean_time_acc;

    set < double,
        boost::accumulators::features< boost::accumulators::tag::moment<2> > >
        moment_time_acc;
   
    set < double,
        boost::accumulators::features< boost::accumulators::tag::variance> >
        variance_time_acc;
};

inline void count_time_stats::push_back(count_and_time_data_point const& x)
{
    lock
        nt(mean_time_mtx),
        tt(moment_time_mtx),
        et(variance_time_mtx),

    ++count_time_stats_size;
    
    mean_time_acc(x.time);
    moment_time_acc(x.time);
    variance_time_acc(x.time);
}

inline boost::int64_t count_time_stats::size() const
{
    return count_time_stats_size.load();
}

inline double count_time_stats::mean_time() const
{
    lock nt(mean_time_mtx);
    return boost::accumulators::extract::mean(mean_time_acc);
}   
   
inline double count_time_stats::moment_time() const
{
    lock tt(moment_time_mtx);
    return boost::accumulators::extract::moment<2>(moment_time_acc);
}

inline double count_time_stats::variance_time() const
{
    lock vt(variance_time_mtx);
    return boost::accumulators::extract::variance(variance_time_acc);
}

}}}

#endif // HPX_05A1C29B_DB73_463A_8C9D_B8EDC3B69F5E

