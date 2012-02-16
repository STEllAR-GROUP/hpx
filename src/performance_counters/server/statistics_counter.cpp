//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/components/derived_component_factory.hpp>
#include <hpx/runtime/actions/continuation.hpp>
#include <hpx/runtime/agas/interface.hpp>
#include <hpx/performance_counters/counters.hpp>
#include <hpx/performance_counters/counter_creators.hpp>
#include <hpx/performance_counters/stubs/performance_counter.hpp>
#include <hpx/performance_counters/server/statistics_counter.hpp>

#include <boost/version.hpp>
#include <boost/chrono/chrono.hpp>
#include <boost/accumulators/statistics/mean.hpp>
#include <boost/accumulators/statistics/max.hpp>
#include <boost/accumulators/statistics/min.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace performance_counters { namespace server
{
    namespace detail
    {
        template <typename Statistic>
        struct counter_type_from_statistic;

        template <>
        struct counter_type_from_statistic<boost::accumulators::tag::mean>
        {
            typedef boost::accumulators::tag::mean statistics_tag;
            typedef boost::accumulators::accumulator_set<
                double, boost::accumulators::stats<statistics_tag>
            > accumulator_type;

            enum { value = counter_average_count };

            static boost::uint64_t call(accumulator_type& accum)
            {
                return static_cast<boost::uint64_t>(
                    boost::accumulators::mean(accum));
            }
        };

        template <>
        struct counter_type_from_statistic<boost::accumulators::tag::max>
        {
            typedef boost::accumulators::tag::max statistics_tag;
            typedef boost::accumulators::accumulator_set<
                double, boost::accumulators::stats<statistics_tag>
            > accumulator_type;

            enum { value = counter_statistics_max };

            static boost::uint64_t call(accumulator_type& accum)
            {
                return static_cast<boost::uint64_t>(
                    (boost::accumulators::max)(accum));
            }
        };

        template <>
        struct counter_type_from_statistic<boost::accumulators::tag::min>
        {
            typedef boost::accumulators::tag::min statistics_tag;
            typedef boost::accumulators::accumulator_set<
                double, boost::accumulators::stats<statistics_tag>
            > accumulator_type;

            enum { value = counter_statistics_min };

            static boost::uint64_t call(accumulator_type& accum)
            {
                return static_cast<boost::uint64_t>(
                    (boost::accumulators::min)(accum));
            }
        };
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename Statistic>
    statistics_counter<Statistic>::statistics_counter(
            counter_info const& info, std::string const& base_counter_name,
            std::size_t base_time_interval)
      : base_type_holder(info),
        timer_(boost::bind(&statistics_counter::evaluate, this),
            1000 * base_time_interval, info.fullname_, true),
        base_counter_name_(ensure_counter_prefix(base_counter_name))
    {
        if (base_time_interval == 0) {
            HPX_THROW_EXCEPTION(bad_parameter,
                "statistics_counter<Statistic>::statistics_counter",
                "base interval is specified to be zero");
        }

        boost::uint64_t type
            = detail::counter_type_from_statistic<Statistic>::value;

        if (info.type_ != type) {
            HPX_THROW_EXCEPTION(bad_parameter,
                "average_count_counter<Statistic>::statistics_counter",
                "unexpected counter type specified for elapsed_time_counter");
        }

        // acquire the current value of the base counter
        counter_value base_value;
        evaluate_base_counter(base_value);

        {
            mutex_type::scoped_lock l(mtx_);
            value_(static_cast<double>(base_value.value_));
            prev_value_ = base_value;
        }

        timer_.start();       // start interval timer
    }

    template <typename Statistic>
    void statistics_counter<Statistic>::get_counter_value(counter_value& value)
    {
        mutex_type::scoped_lock l(mtx_);

        value = prev_value_;                // return value
        value.value_ = detail::counter_type_from_statistic<Statistic>::call(value_);
        value.status_ = status_new_data;
        value.time_ = boost::chrono::high_resolution_clock::now().
            time_since_epoch().count();

        value_ = mean_accumulator_type();   // reset accumulator
        value_(static_cast<double>(prev_value_.value_));         // start off with last base value
    }

    template <typename Statistic>
    void statistics_counter<Statistic>::evaluate()
    {
        // gather current base value
        counter_value base_value;
        evaluate_base_counter(base_value);

        // simply average the measured base counter values since it got queried
        // for the last time
        counter_value value;
        if (base_value.scaling_ != prev_value_.scaling_ ||
            base_value.scale_inverse_ != prev_value_.scale_inverse_)
        {
            // not supported right now
            HPX_THROW_EXCEPTION(not_implemented,
                "average_count_counter<Statistic>::get_counter_value",
                "base counter should keep scaling constant over time");
        }
        else {
            mutex_type::scoped_lock l(mtx_);
            value_(static_cast<double>(base_value.value_));          // accumulate new value
        }
    }

    template <typename Statistic>
    void statistics_counter<Statistic>::evaluate_base_counter(
        counter_value& value)
    {
        {
            // lock here to avoid checking out multiple reference counted GIDs
            // from AGAS
            mutex_type::scoped_lock l(mtx_);

            if (!base_counter_id_) {
                // get or create the base counter
                error_code ec;
                base_counter_id_ = get_counter(base_counter_name_, ec);
                if (HPX_UNLIKELY(ec || !base_counter_id_))
                {
                    // base counter could not be retrieved
                    HPX_THROW_EXCEPTION(bad_parameter,
                        "average_count_counter<Statistic>::evaluate_base_counter",
                        boost::str(
                            boost::format("could not get or create performance counter: '%s'") %
                                base_counter_name_)
                        )
                }
            }
        }

        // query the actual value
        value = stubs::performance_counter::get_value(base_counter_id_);
    }
}}}

///////////////////////////////////////////////////////////////////////////////
// Average
typedef hpx::components::managed_component<
    hpx::performance_counters::server::statistics_counter<
        boost::accumulators::tag::mean>
> average_count_counter_type;

template HPX_EXPORT class hpx::performance_counters::server::statistics_counter<
    boost::accumulators::tag::mean>;
template HPX_EXPORT class hpx::performance_counters::server::statistics_counter<
    boost::accumulators::tag::max>;
template HPX_EXPORT class hpx::performance_counters::server::statistics_counter<
    boost::accumulators::tag::min>;

HPX_REGISTER_DERIVED_COMPONENT_FACTORY_EX(
    average_count_counter_type, average_count_counter,
    "base_performance_counter", true);
HPX_DEFINE_GET_COMPONENT_TYPE(average_count_counter_type::wrapped_type);

///////////////////////////////////////////////////////////////////////////////
// Max
typedef hpx::components::managed_component<
    hpx::performance_counters::server::statistics_counter<
        boost::accumulators::tag::max>
> max_count_counter_type;

HPX_REGISTER_DERIVED_COMPONENT_FACTORY_EX(
    max_count_counter_type, max_count_counter,
    "base_performance_counter", true);
HPX_DEFINE_GET_COMPONENT_TYPE(max_count_counter_type::wrapped_type);

///////////////////////////////////////////////////////////////////////////////
// Min
typedef hpx::components::managed_component<
    hpx::performance_counters::server::statistics_counter<
        boost::accumulators::tag::min>
> min_count_counter_type;

HPX_REGISTER_DERIVED_COMPONENT_FACTORY_EX(
    min_count_counter_type, min_count_counter,
    "base_performance_counter", true);
HPX_DEFINE_GET_COMPONENT_TYPE(min_count_counter_type::wrapped_type);

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace performance_counters { namespace detail
{
    /// Creation function for statistics performance counters to be registered
    /// with the counter types.
    naming::id_type statistics_counter_creator(counter_info const& info,
        error_code& ec)
    {
        switch (info.type_) {
        case counter_average_count:
        case counter_statistics_max:
        case counter_statistics_min:
            {
                counter_path_elements paths;
                get_counter_path_elements(info.fullname_, paths, ec);
                if (ec) return naming::invalid_id;

                if (!paths.parentinstance_is_basename_) {
                    HPX_THROWS_IF(ec, bad_parameter,
                        "statistics_counter_creator", "invalid aggregate counter "
                            "name (instance name must be valid base counter name)");
                    return naming::invalid_id;
                }

                std::string base_name;
                get_counter_name(paths.parentinstancename_, base_name, ec);
                if (ec) return naming::invalid_id;

                std::size_t interval = 1000;
                if (!paths.parameters_.empty()) {
                    // try to interpret the additional parameter as interval
                    // time (ms)
                    try {
                        interval = boost::lexical_cast<std::size_t>(paths.parameters_);
                    }
                    catch (boost::bad_lexical_cast const& e) {
                        HPX_THROWS_IF(ec, bad_parameter,
                            "statistics_counter_creator", e.what());
                        return naming::invalid_id;
                    }
                }
                return create_statistics_counter(info, base_name, interval, ec);
            }
            break;

        default:
            HPX_THROWS_IF(ec, bad_parameter, "statistics_counter_creator",
                "invalid counter type requested");
            return naming::invalid_id;
        }
    }
}}}

