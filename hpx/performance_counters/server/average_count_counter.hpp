//  Copyright (c) 2007-2011 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PERFORMANCE_COUNTERS_SERVER_AVERAGE_COUNT_COUNTER_SEP_30_2011_1045AM)
#define HPX_PERFORMANCE_COUNTERS_SERVER_AVERAGE_COUNT_COUNTER_SEP_30_2011_1045AM

#include <hpx/hpx_fwd.hpp>
#include <hpx/performance_counters/server/base_performance_counter.hpp>
#include <hpx/util/interval_timer.hpp>
#include <hpx/util/spinlock.hpp>

#include <boost/accumulators/statistics/mean.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace performance_counters { namespace server
{
    // This counter exposes the average count of items processed during the 
    // given base time interval. The counter relies on querying a steadily 
    // growing counter value.
    class HPX_EXPORT average_count_counter 
      : public base_performance_counter,
        public components::managed_component_base<average_count_counter> 
    {
        typedef components::managed_component_base<average_count_counter> base_type;

    public:
        typedef average_count_counter type_holder;
        typedef base_performance_counter base_type_holder;

        average_count_counter() {}
        average_count_counter(counter_info const& info,
            std::string const& base_counter_name, 
            std::size_t base_time_interval);

        void get_counter_value(counter_value& value);

        /// \brief finalize() will be called just before the instance gets 
        ///        destructed
        void finalize() 
        {
            base_performance_counter::finalize();
            base_type::finalize();
        }

        static components::component_type get_component_type() 
        { 
            return base_type::get_component_type(); 
        }
        static void set_component_type(components::component_type t) 
        { 
            base_type::set_component_type(t);
        }

    protected:
        void evaluate_base_counter(counter_value& value);
        void evaluate();

    private:
        typedef util::spinlock mutex_type;
        mutable mutex_type mtx_;

        hpx::util::interval_timer timer_; ///< base time interval in milliseconds
        std::string base_counter_name_;   ///< name of base counter to be queried
        naming::id_type base_counter_id_;
        typedef boost::accumulators::accumulator_set<
            boost::int64_t, 
            boost::accumulators::stats<boost::accumulators::tag::mean> 
        > mean_accumulator_type;
        mean_accumulator_type value_;
        counter_value prev_value_;
    };
}}}

#endif
