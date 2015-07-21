//  Copyright (c) 2007-2013 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/components/derived_component_factory.hpp>
#include <hpx/runtime/actions/continuation.hpp>
#include <hpx/runtime/agas/interface.hpp>
#include <hpx/util/high_resolution_clock.hpp>
#include <hpx/performance_counters/counters.hpp>
#include <hpx/performance_counters/counter_creators.hpp>
#include <hpx/performance_counters/stubs/performance_counter.hpp>
#include <hpx/performance_counters/server/statistics_counter.hpp>

#include <boost/version.hpp>
#include <boost/chrono/chrono.hpp>
#include <boost/thread/locks.hpp>

#define BOOST_SPIRIT_USE_PHOENIX_V3
#include <boost/spirit/include/qi_char.hpp>
#include <boost/spirit/include/qi_numeric.hpp>
#include <boost/spirit/include/qi_operator.hpp>
#include <boost/spirit/include/qi_parse.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace performance_counters { namespace server
{
    ///////////////////////////////////////////////////////////////////////////
    template <typename Statistic>
    statistics_counter<Statistic>::statistics_counter(
            counter_info const& info, std::string const& base_counter_name,
            boost::uint64_t parameter1, boost::uint64_t parameter2)
      : base_type_holder(info),
        timer_(boost::bind(&statistics_counter::evaluate, this_()),
            boost::bind(&statistics_counter::on_terminate, this_()),
            1000 * parameter1, info.fullname_, true),
        base_counter_name_(ensure_counter_prefix(base_counter_name)),
        value_(detail::counter_type_from_statistic<Statistic>::create(parameter2)),
        parameter1_(parameter1), parameter2_(parameter2)
    {
        if (parameter1 == 0) {
            HPX_THROW_EXCEPTION(bad_parameter,
                "statistics_counter<Statistic>::statistics_counter",
                "base interval is specified to be zero");
        }

        if (info.type_ != counter_aggregating) {
            HPX_THROW_EXCEPTION(bad_parameter,
                "statistics_counter<Statistic>::statistics_counter",
                "unexpected counter type specified");
        }
    }

    template <typename Statistic>
    hpx::performance_counters::counter_value
        statistics_counter<Statistic>::get_counter_value(bool reset)
    {
        boost::lock_guard<mutex_type> l(mtx_);

        hpx::performance_counters::counter_value value;

        prev_value_.value_ = detail::counter_type_from_statistic<Statistic>::call(*value_);
        prev_value_.status_ = status_new_data;
        prev_value_.time_ = static_cast<boost::int64_t>(hpx::get_system_uptime());
        prev_value_.count_ = ++invocation_count_;
        value = prev_value_;                              // return value

        if (reset || detail::counter_type_from_statistic<Statistic>::need_reset::value)
        {
            value_.reset(detail::counter_type_from_statistic<Statistic>::create(
                parameter2_)); // reset accumulator
            (*value_)(static_cast<double>(prev_value_.value_));  // start off with last base value
        }

        return value;
    }

    template <typename Statistic>
    bool statistics_counter<Statistic>::evaluate()
    {
        // gather current base value
        counter_value base_value;
        if (!evaluate_base_counter(base_value))
            return false;

        // simply average the measured base counter values since it got queried
        // for the last time
        counter_value value;
        if (base_value.scaling_ != prev_value_.scaling_ ||
            base_value.scale_inverse_ != prev_value_.scale_inverse_)
        {
            // not supported right now
            HPX_THROW_EXCEPTION(not_implemented,
                "statistics_counter<Statistic>::evaluate",
                "base counter should keep scaling constant over time");
            return false;
        }
        else {
            boost::lock_guard<mutex_type> l(mtx_);
            (*value_)(static_cast<double>(base_value.value_));          // accumulate new value
        }
        return true;
    }

    template <typename Statistic>
    bool statistics_counter<Statistic>::ensure_base_counter()
    {
        // lock here to avoid checking out multiple reference counted GIDs
        // from AGAS
        boost::lock_guard<mutex_type> l(mtx_);

        if (!base_counter_id_) {
            // get or create the base counter
            error_code ec(lightweight);
            base_counter_id_ = get_counter(base_counter_name_, ec);
            if (HPX_UNLIKELY(ec || !base_counter_id_))
            {
                // base counter could not be retrieved
                HPX_THROW_EXCEPTION(bad_parameter,
                    "statistics_counter<Statistic>::evaluate_base_counter",
                    boost::str(boost::format(
                        "could not get or create performance counter: '%s'") %
                            base_counter_name_)
                    )
                return false;
            }
        }

        return true;
    }

    template <typename Statistic>
    bool statistics_counter<Statistic>::evaluate_base_counter(
        counter_value& value)
    {
        // query the actual value
        if (!base_counter_id_ && !ensure_base_counter())
            return false;

        value = stubs::performance_counter::get_value(base_counter_id_);
        return true;
    }

    ///////////////////////////////////////////////////////////////////////////
    // Start and stop this counter. We dispatch the calls to the base counter
    // and control our own interval_timer.
    template <typename Statistic>
    bool statistics_counter<Statistic>::start()
    {
        if (!timer_.is_started()) {
            // start base counter
            if (!base_counter_id_ && !ensure_base_counter())
                return false;

            bool result = stubs::performance_counter::start(base_counter_id_);
            if (result) {
                // acquire the current value of the base counter
                counter_value base_value;
                if (evaluate_base_counter(base_value))
                {
                    boost::lock_guard<mutex_type> l(mtx_);
                    (*value_)(static_cast<double>(base_value.value_));
                    prev_value_ = base_value;
                }

                // start counter
                timer_.start();
            }
            return result;
        }
        return false;
    }

    template <typename Statistic>
    bool statistics_counter<Statistic>::stop()
    {
        if (timer_.is_started()) {
            timer_.stop();

            if (!base_counter_id_ && !ensure_base_counter())
                return false;
            return stubs::performance_counter::stop(base_counter_id_);
        }
        return false;
    }

    template <typename Statistic>
    void statistics_counter<Statistic>::reset_counter_value()
    {
        boost::lock_guard<mutex_type> l(mtx_);

        value_.reset(detail::counter_type_from_statistic<Statistic>::create(
            parameter2_)); // reset accumulator
        (*value_)(static_cast<double>(prev_value_.value_));  // start off with last base value
    }
}}}

///////////////////////////////////////////////////////////////////////////////
template class HPX_EXPORT hpx::performance_counters::server::statistics_counter<
    boost::accumulators::tag::mean>;
template class HPX_EXPORT hpx::performance_counters::server::statistics_counter<
    boost::accumulators::tag::variance>;
template class HPX_EXPORT hpx::performance_counters::server::statistics_counter<
    boost::accumulators::tag::median>;
template class HPX_EXPORT hpx::performance_counters::server::statistics_counter<
    boost::accumulators::tag::rolling_mean>;
template class HPX_EXPORT hpx::performance_counters::server::statistics_counter<
    boost::accumulators::tag::max>;
template class HPX_EXPORT hpx::performance_counters::server::statistics_counter<
    boost::accumulators::tag::min>;

///////////////////////////////////////////////////////////////////////////////
// Average
typedef hpx::components::managed_component<
    hpx::performance_counters::server::statistics_counter<
        boost::accumulators::tag::mean>
> average_count_counter_type;

HPX_REGISTER_DERIVED_COMPONENT_FACTORY(
    average_count_counter_type, average_count_counter,
    "base_performance_counter", hpx::components::factory_enabled)
HPX_DEFINE_GET_COMPONENT_TYPE(average_count_counter_type::wrapped_type)

///////////////////////////////////////////////////////////////////////////////
// Variance
typedef hpx::components::managed_component<
    hpx::performance_counters::server::statistics_counter<
        boost::accumulators::tag::variance>
> variance_count_counter_type;

HPX_REGISTER_DERIVED_COMPONENT_FACTORY(
    variance_count_counter_type, variance_count_counter,
    "base_performance_counter", hpx::components::factory_enabled)
HPX_DEFINE_GET_COMPONENT_TYPE(variance_count_counter_type::wrapped_type)

///////////////////////////////////////////////////////////////////////////////
// Rooling average
typedef hpx::components::managed_component<
    hpx::performance_counters::server::statistics_counter<
        boost::accumulators::tag::rolling_mean>
> rolling_mean_count_counter_type;

HPX_REGISTER_DERIVED_COMPONENT_FACTORY(
    rolling_mean_count_counter_type, rolling_mean_count_counter,
    "base_performance_counter", hpx::components::factory_enabled)
HPX_DEFINE_GET_COMPONENT_TYPE(rolling_mean_count_counter_type::wrapped_type)

///////////////////////////////////////////////////////////////////////////////
// Median
typedef hpx::components::managed_component<
    hpx::performance_counters::server::statistics_counter<
        boost::accumulators::tag::median>
> median_count_counter_type;

HPX_REGISTER_DERIVED_COMPONENT_FACTORY(
    median_count_counter_type, median_count_counter,
    "base_performance_counter", hpx::components::factory_enabled)
HPX_DEFINE_GET_COMPONENT_TYPE(median_count_counter_type::wrapped_type)

///////////////////////////////////////////////////////////////////////////////
// Max
typedef hpx::components::managed_component<
    hpx::performance_counters::server::statistics_counter<
        boost::accumulators::tag::max>
> max_count_counter_type;

HPX_REGISTER_DERIVED_COMPONENT_FACTORY(
    max_count_counter_type, max_count_counter,
    "base_performance_counter", hpx::components::factory_enabled)
HPX_DEFINE_GET_COMPONENT_TYPE(max_count_counter_type::wrapped_type)

///////////////////////////////////////////////////////////////////////////////
// Min
typedef hpx::components::managed_component<
    hpx::performance_counters::server::statistics_counter<
        boost::accumulators::tag::min>
> min_count_counter_type;

HPX_REGISTER_DERIVED_COMPONENT_FACTORY(
    min_count_counter_type, min_count_counter,
    "base_performance_counter", hpx::components::factory_enabled)
HPX_DEFINE_GET_COMPONENT_TYPE(min_count_counter_type::wrapped_type)

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace performance_counters { namespace detail
{
    /// Creation function for aggregating performance counters to be registered
    /// with the counter types.
    naming::gid_type statistics_counter_creator(counter_info const& info,
        error_code& ec)
    {
        switch (info.type_) {
        case counter_aggregating:
            {
                counter_path_elements paths;
                get_counter_path_elements(info.fullname_, paths, ec);
                if (ec) return naming::invalid_gid;

                if (!paths.parentinstance_is_basename_) {
                    HPX_THROWS_IF(ec, bad_parameter,
                        "statistics_counter_creator", "invalid aggregate counter "
                            "name (instance name must be valid base counter name)");
                    return naming::invalid_gid;
                }

                std::string base_name;
                get_counter_name(paths.parentinstancename_, base_name, ec);
                if (ec) return naming::invalid_gid;

                std::vector<boost::int64_t> parameters;
                if (!paths.parameters_.empty()) {
                    // try to interpret the additional parameter as interval
                    // time (ms)
                    namespace qi = boost::spirit::qi;
                    if (!qi::parse(paths.parameters_.begin(), paths.parameters_.end(),
                            qi::int_ % ',', parameters))
                    {
                        HPX_THROWS_IF(ec, bad_parameter,
                            "statistics_counter_creator",
                            "invalid parameter specification for counter: " +
                                paths.parameters_);
                        return naming::invalid_gid;
                    }
                }
                else {
                    parameters.push_back(1000);       // sample interval
                    parameters.push_back(10);         // rolling window
                }
                return create_statistics_counter(info, base_name, parameters, ec);
            }
            break;

        default:
            HPX_THROWS_IF(ec, bad_parameter, "statistics_counter_creator",
                "invalid counter type requested");
            return naming::invalid_gid;
        }
    }
}}}

