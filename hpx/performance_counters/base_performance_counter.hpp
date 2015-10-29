//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PERFORMANCE_COUNTERS_BASE_PERFORMANCE_COUNTER_JAN_18_2013_1036AM)
#define HPX_PERFORMANCE_COUNTERS_BASE_PERFORMANCE_COUNTER_JAN_18_2013_1036AM

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/components/component_type.hpp>
#include <hpx/runtime/components/server/component_base.hpp>
#include <hpx/runtime/actions/component_action.hpp>
#include <hpx/performance_counters/counters.hpp>
#include <hpx/performance_counters/server/base_performance_counter.hpp>

///////////////////////////////////////////////////////////////////////////////
//[performance_counter_base_class
namespace hpx { namespace performance_counters
{
    template <typename Derived>
    class base_performance_counter;
}}
//]

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace performance_counters
{
    template <typename Derived>
    class base_performance_counter
      : public hpx::performance_counters::server::base_performance_counter,
        public hpx::components::component_base<Derived>
    {
    private:
        typedef hpx::components::component_base<Derived> base_type;

    public:
        typedef Derived type_holder;
        typedef hpx::performance_counters::server::base_performance_counter
            base_type_holder;

        base_performance_counter()
        {}

        base_performance_counter(hpx::performance_counters::counter_info const& info)
          : base_type_holder(info)
        {}

        // Disambiguate finalize() which is implemented in both base classes
        void finalize()
        {
            base_type_holder::finalize();
            base_type::finalize();
        }

        using base_type::get_component_type;
        using base_type::set_component_type;
    };
}}

#endif
