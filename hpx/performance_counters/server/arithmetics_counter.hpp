//  Copyright (c) 2007-2013 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PERFORMANCE_COUNTERS_SERVER_ARITHMETICS_COUNTER_APR_10_2013_1002AM)
#define HPX_PERFORMANCE_COUNTERS_SERVER_ARITHMETICS_COUNTER_APR_10_2013_1002AM

#include <hpx/config.hpp>
#include <hpx/runtime/components/server/component_base.hpp>
#include <hpx/performance_counters/server/base_performance_counter.hpp>
#include <hpx/util/interval_timer.hpp>
#include <hpx/lcos/local/spinlock.hpp>

#include <string>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace performance_counters { namespace server
{
    ///////////////////////////////////////////////////////////////////////////
    // This counter exposes the result of an arithmetic operation The counter
    // relies on querying two base counters.
    template <typename Operation>
    class arithmetics_counter
      : public base_performance_counter,
        public components::component_base<
            arithmetics_counter<Operation> >
    {
        typedef components::component_base<
            arithmetics_counter<Operation> > base_type;

    public:
        typedef arithmetics_counter type_holder;
        typedef base_performance_counter base_type_holder;

        arithmetics_counter()
          : mtx_(), invocation_count_(0)
        {}

        arithmetics_counter(counter_info const& info,
            std::vector<std::string> const& base_counter_names);

        /// Overloads from the base_counter base class.
        hpx::performance_counters::counter_value
            get_counter_value(bool reset = false);

        bool start();
        bool stop();
        void reset_counter_value();

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
        bool evaluate_base_counter(naming::id_type& base_counter_id,
            std::string const& name, counter_value& value);
        bool ensure_base_counter(naming::id_type& base_counter_id,
            std::string const& name);

    private:
        typedef lcos::local::spinlock mutex_type;
        mutable mutex_type mtx_;

        std::vector<std::string> base_counter_names_;
        ///^ names of base counters to be queried
        std::vector<naming::id_type> base_counter_ids_;
        boost::uint64_t invocation_count_;
    };
}}}

#endif
