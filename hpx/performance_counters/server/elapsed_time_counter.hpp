//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PERFORMANCE_COUNTERS_SERVER_ELAPSED_TIME_COUNTER_SEP_18_2011_1133AM)
#define HPX_PERFORMANCE_COUNTERS_SERVER_ELAPSED_TIME_COUNTER_SEP_18_2011_1133AM

#include <hpx/config.hpp>
#include <hpx/performance_counters/server/base_performance_counter.hpp>
#include <hpx/runtime/components/server/component_base.hpp>
#include <hpx/util/high_resolution_timer.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace performance_counters { namespace server
{
    class HPX_EXPORT elapsed_time_counter
      : public base_performance_counter,
        public components::component_base<elapsed_time_counter>
    {
        typedef components::component_base<elapsed_time_counter> base_type;

    public:
        typedef elapsed_time_counter type_holder;
        typedef base_performance_counter base_type_holder;

        elapsed_time_counter() {}
        elapsed_time_counter(counter_info const& info);

        hpx::performance_counters::counter_value get_counter_value(bool reset);
        void reset_counter_value();

        bool start() { return false; }
        bool stop() { return false; }

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
    };
}}}

#endif
