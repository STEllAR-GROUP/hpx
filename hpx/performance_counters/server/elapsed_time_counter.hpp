//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PERFORMANCE_COUNTERS_SERVER_ELAPSED_TIME_COUNTER_SEP_18_2011_1133AM)
#define HPX_PERFORMANCE_COUNTERS_SERVER_ELAPSED_TIME_COUNTER_SEP_18_2011_1133AM

#include <hpx/hpx_fwd.hpp>
#include <hpx/performance_counters/server/base_performance_counter.hpp>
#include <hpx/util/high_resolution_timer.hpp>

#include <boost/detail/atomic_count.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace performance_counters { namespace server
{
    class HPX_EXPORT elapsed_time_counter
      : public base_performance_counter,
        public components::managed_component_base<elapsed_time_counter>
    {
        typedef components::managed_component_base<elapsed_time_counter> base_type;

    public:
        typedef elapsed_time_counter type_holder;
        typedef base_performance_counter base_type_holder;

        elapsed_time_counter() {}
        elapsed_time_counter(counter_info const& info);

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

    private:
        boost::int64_t started_at_;
    };
}}}

#endif
