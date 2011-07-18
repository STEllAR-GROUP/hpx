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
      gatherer_size(0),
      point(0,0,0,naming::invalid_gid)
    {}

    boost::int64_t size() const;
    void push_back(data_point const& x);

    double mean_time() const;
    double mean_byte() const;
    double mean_time_per_byte() const;

    double moment_time() const;
    double moment_byte() const;
    double moment_time_per_byte() const;

    boost::int64_t total_bytes() const;
    double total_time() const;    

    private:
    util::high_resolution_timer timer;
    boost::int64_t byte_count;
    boost::int64_t gatherer_size;
    hpx::performance_counters::parcels::data_point point;

    // Create mutexes for accumulator functions.
    mutable mutex_type mean_time_mtx;
    mutable mutex_type mean_byte_mtx;
    mutable mutex_type moment_time_mtx;
    mutable mutex_type moment_byte_mtx;
    mutable mutex_type moment_time_per_byte_mtx;
    mutable mutex_type total_bytes_mtx;
    mutable mutex_type size_mtx;    

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

};

inline void gatherer::push_back(data_point const& x)
{
    lock
        nt(mean_time_mtx),
        nb(mean_byte_mtx),
        tt(moment_time_mtx),
        tb(moment_byte_mtx),
        bb(total_bytes_mtx),
        mb(moment_time_per_byte_mtx),
        sz(size_mtx);
   
    point = x;
    byte_count += point.bytes;
    ++gatherer_size;
    
    mean_time_acc(point.elapsed());
    moment_time_acc(point.elapsed());
    mean_byte_acc(double(point.bytes));
    moment_byte_acc(double(point.bytes));
    BOOST_ASSERT(point.bytes != 0);  
    moment_time_per_byte_acc(point.elapsed() / double(point.bytes));
}

inline boost::int64_t gatherer::size() const
{
    lock sz(size_mtx);
    return gatherer_size;
}

inline double gatherer::mean_time() const
{
    lock nt(mean_time_mtx);
    return boost::accumulators::extract::mean(mean_time_acc);
}   
   
inline double gatherer::moment_time() const
{
    lock tt(moment_time_mtx);
    return boost::accumulators::extract::moment<2>(moment_time_acc);
}

inline double gatherer::mean_byte() const
{
    lock nb(mean_byte_mtx); 
    return boost::accumulators::extract::mean(mean_byte_acc);
}

inline double gatherer::moment_byte() const
{
    lock tb(moment_byte_mtx);
    return boost::accumulators::extract::moment<2>(moment_byte_acc);
}

inline double gatherer::total_time() const
{
    return timer.elapsed();
}

inline boost::int64_t gatherer::total_bytes() const
{
    lock bb(total_bytes_mtx);
    return byte_count;
}

inline double gatherer::mean_time_per_byte() const
{
   lock bb(total_bytes_mtx);
   return total_time() / double(total_bytes());
}

inline double gatherer::moment_time_per_byte() const
{
    lock
        mb(moment_time_per_byte_mtx),
        bb(total_bytes_mtx);

    return boost::accumulators::extract::moment<2>(moment_time_per_byte_acc);
}
}}}

#endif // HPX_05A1C29B_DB73_463A_8C9D_B8EDC3B69F5E

