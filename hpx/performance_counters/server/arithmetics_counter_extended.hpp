//  Copyright (c) 2007-2017 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PERFORMANCE_COUNTERS_SERVER_ARITHMETICS_COUNTER_EXTENDED)
#define HPX_PERFORMANCE_COUNTERS_SERVER_ARITHMETICS_COUNTER_EXTENDED

#include <hpx/config.hpp>
#include <hpx/performance_counters/performance_counter_set.hpp>
#include <hpx/performance_counters/server/base_performance_counter.hpp>
#include <hpx/runtime/components/server/component_base.hpp>

#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace performance_counters { namespace server
{
    ///////////////////////////////////////////////////////////////////////////
    // This counter exposes the result of an arithmetic operation The counter
    // relies on querying two base counters.
    template <typename Statistic>
    class arithmetics_counter_extended
      : public base_performance_counter,
        public components::component_base<
            arithmetics_counter_extended<Statistic> >
    {
        typedef components::component_base<
            arithmetics_counter_extended<Statistic> > base_type;

    public:
        typedef arithmetics_counter_extended type_holder;
        typedef base_performance_counter base_type_holder;

        arithmetics_counter_extended() = default;

        arithmetics_counter_extended(counter_info const& info,
            std::vector<std::string> const& base_counter_names);

        /// Overloads from the base_counter base class.
        hpx::performance_counters::counter_value
            get_counter_value(bool reset = false);

        bool start();
        bool stop();
        void reset_counter_value();

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
        // base counters to be queried
        performance_counter_set counters_;
    };
}}}

#endif
