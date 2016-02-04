//  Copyright (c) 2007-2015 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PERFORMANCE_COUNTERS_SERVER_AVERAGE_COUNT_COUNTER_SEP_30_2011_1045AM)
#define HPX_PERFORMANCE_COUNTERS_SERVER_AVERAGE_COUNT_COUNTER_SEP_30_2011_1045AM

#include <hpx/config.hpp>
#include <hpx/runtime/naming/id_type.hpp>
#include <hpx/runtime/components/server/component_base.hpp>
#include <hpx/performance_counters/server/base_performance_counter.hpp>
#include <hpx/util/interval_timer.hpp>
#include <hpx/lcos/local/spinlock.hpp>

#include <boost/smart_ptr/scoped_ptr.hpp>
#include <boost/cstdint.hpp>

#include <string>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace performance_counters { namespace server
{
    namespace detail
    {
        struct counter_type_from_statistic_base
        {
            virtual ~counter_type_from_statistic_base() {}

            virtual bool need_reset() const = 0;
            virtual double get_value() = 0;
            virtual void add_value(double value) = 0;
        };
    }

    ///////////////////////////////////////////////////////////////////////////
    // This counter exposes the average count of items processed during the
    // given base time interval. The counter relies on querying a steadily
    // growing counter value.
    template <typename Statistic>
    class statistics_counter
      : public base_performance_counter,
        public components::component_base<statistics_counter<Statistic> >
    {
        typedef components::component_base<
            statistics_counter<Statistic> > base_type;

        // avoid warnings about using this in member initializer list
        statistics_counter* this_() { return this; }

    public:
        typedef statistics_counter type_holder;
        typedef base_performance_counter base_type_holder;

        statistics_counter() {}

        statistics_counter(counter_info const& info,
            std::string const& base_counter_name,
            boost::uint64_t parameter1, boost::uint64_t parameter2);

        /// Overloads from the base_counter base class.
        hpx::performance_counters::counter_value
            get_counter_value(bool reset = false);

        bool start();

        bool stop();

        void reset_counter_value();

        void on_terminate() {}

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
        bool evaluate_base_counter(counter_value& value);
        bool evaluate();
        bool ensure_base_counter();

    private:
        typedef lcos::local::spinlock mutex_type;
        mutable mutex_type mtx_;

        hpx::util::interval_timer timer_; ///< base time interval in milliseconds
        std::string base_counter_name_;   ///< name of base counter to be queried
        naming::id_type base_counter_id_;

        boost::scoped_ptr<detail::counter_type_from_statistic_base> value_;
        counter_value prev_value_;

        boost::uint64_t parameter1_, parameter2_;
    };
}}}

#endif
