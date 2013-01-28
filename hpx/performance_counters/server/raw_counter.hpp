//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PERFORMANCE_COUNTERS_SERVER_RAW_COUNTER_MAR_03_2009_0743M)
#define HPX_PERFORMANCE_COUNTERS_SERVER_RAW_COUNTER_MAR_03_2009_0743M

#include <hpx/hpx_fwd.hpp>
#include <hpx/performance_counters/server/base_performance_counter.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace performance_counters { namespace server
{
    class HPX_EXPORT raw_counter
      : public base_performance_counter,
        public components::managed_component_base<raw_counter>
    {
        typedef components::managed_component_base<raw_counter> base_type;

    public:
        typedef raw_counter type_holder;
        typedef base_performance_counter base_type_holder;

        raw_counter() {}
        raw_counter(counter_info const& info, HPX_STD_FUNCTION<boost::int64_t()> f);

        hpx::performance_counters::counter_value get_counter_value();

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
        HPX_STD_FUNCTION<boost::int64_t()> f_;
    };
}}}

#endif
