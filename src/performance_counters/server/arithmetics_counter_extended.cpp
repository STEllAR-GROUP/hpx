//  Copyright (c) 2007-2017 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// make inspect happy: hpxinspect:nodeprecatedname:boost::is_any_of

#include <hpx/config.hpp>
#include <hpx/runtime/runtime_fwd.hpp>
#include <hpx/runtime/components/derived_component_factory.hpp>
#include <hpx/runtime/launch_policy.hpp>
#include <hpx/performance_counters/counters.hpp>
#include <hpx/performance_counters/counter_creators.hpp>
#include <hpx/performance_counters/stubs/performance_counter.hpp>
#include <hpx/performance_counters/server/arithmetics_counter_extended.hpp>

#include <boost/algorithm/string.hpp>
#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics/stats.hpp>
#include <boost/accumulators/statistics/mean.hpp>
#include <boost/accumulators/statistics/median.hpp>
#include <boost/accumulators/statistics/variance.hpp>
#include <boost/accumulators/statistics/max.hpp>
#include <boost/accumulators/statistics/min.hpp>

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace performance_counters { namespace server
{
    ///////////////////////////////////////////////////////////////////////////
    template <typename Statistic>
    arithmetics_counter_extended<Statistic>::arithmetics_counter_extended(
            counter_info const& info,
            std::vector<std::string> const& base_counter_names)
      : base_type_holder(info),
        counters_(base_counter_names)
    {
        if (info.type_ != counter_aggregating) {
            HPX_THROW_EXCEPTION(bad_parameter,
                "arithmetics_counter_extended<Statistic>::"
                    "arithmetics_counter_extended",
                "unexpected counter type specified");
        }
    }

    namespace detail
    {
        template <typename Statistic>
        struct statistic_get_value;

        template <>
        struct statistic_get_value<boost::accumulators::tag::mean>
        {
            template <typename Accumulator>
            static double call(Accumulator const& accum)
            {
                return boost::accumulators::mean(accum);
            }
        };

        template <>
        struct statistic_get_value<boost::accumulators::tag::variance>
        {
            template <typename Accumulator>
            static double call(Accumulator const& accum)
            {
                return boost::accumulators::variance(accum);
            }
        };

        template <>
        struct statistic_get_value<boost::accumulators::tag::median>
        {
            template <typename Accumulator>
            static double call(Accumulator const& accum)
            {
                return boost::accumulators::median(accum);
            }
        };

        template <>
        struct statistic_get_value<boost::accumulators::tag::min>
        {
            template <typename Accumulator>
            static double call(Accumulator const& accum)
            {
                return (boost::accumulators::min)(accum);
            }
        };

        template <>
        struct statistic_get_value<boost::accumulators::tag::max>
        {
            template <typename Accumulator>
            static double call(Accumulator const& accum)
            {
                return (boost::accumulators::max)(accum);
            }
        };
    }

    template <typename Statistic>
    hpx::performance_counters::counter_value
        arithmetics_counter_extended<Statistic>::get_counter_value(bool reset)
    {
        std::vector<counter_value> base_values =
            counters_.get_counter_values(hpx::launch::sync);

        // apply arithmetic Statistic
        typedef boost::accumulators::accumulator_set<
            double, boost::accumulators::stats<Statistic>
        > accumulator_type;

        accumulator_type accum;
        for (counter_value const& base_value : base_values)
        {
            accum(base_value.get_value<double>());
        }
        double value = detail::statistic_get_value<Statistic>::call(accum);

        if (base_values[0].scale_inverse_ && base_values[0].scaling_ != 1.0) //-V550
        {
            base_values[0].value_ =
                static_cast<std::int64_t>(value * base_values[0].scaling_);
        }
        else {
            base_values[0].value_ =
                static_cast<std::int64_t>(value / base_values[0].scaling_);
        }

        base_values[0].time_ = static_cast<std::int64_t>(hpx::get_system_uptime());
        base_values[0].count_ = counters_.get_invocation_count();

        return base_values[0];
    }

    template <typename Statistic>
    bool arithmetics_counter_extended<Statistic>::start()
    {
        return counters_.start(hpx::launch::sync);
    }

    template <typename Statistic>
    bool arithmetics_counter_extended<Statistic>::stop()
    {
        return counters_.stop(hpx::launch::sync);
    }

    template <typename Statistic>
    void arithmetics_counter_extended<Statistic>::reset_counter_value()
    {
        counters_.reset(hpx::launch::sync);
    }
}}}

///////////////////////////////////////////////////////////////////////////////
template class HPX_EXPORT
hpx::performance_counters::server::arithmetics_counter_extended<
    boost::accumulators::tag::mean>;
template class HPX_EXPORT
hpx::performance_counters::server::arithmetics_counter_extended<
    boost::accumulators::tag::variance>;
template class HPX_EXPORT
hpx::performance_counters::server::arithmetics_counter_extended<
    boost::accumulators::tag::median>;
template class HPX_EXPORT
hpx::performance_counters::server::arithmetics_counter_extended<
    boost::accumulators::tag::min>;
template class HPX_EXPORT
hpx::performance_counters::server::arithmetics_counter_extended<
    boost::accumulators::tag::max>;

///////////////////////////////////////////////////////////////////////////////
// /arithmetic/mean
typedef hpx::components::component<
    hpx::performance_counters::server::arithmetics_counter_extended<
        boost::accumulators::tag::mean>
> mean_arithmetics_counter_type;

HPX_REGISTER_DERIVED_COMPONENT_FACTORY(
    mean_arithmetics_counter_type, mean_arithmetics_counter,
    "base_performance_counter", hpx::components::factory_enabled)
HPX_DEFINE_GET_COMPONENT_TYPE(mean_arithmetics_counter_type::wrapped_type)

///////////////////////////////////////////////////////////////////////////////
// /arithmetic/variance
typedef hpx::components::component<
    hpx::performance_counters::server::arithmetics_counter_extended<
        boost::accumulators::tag::variance>
> variance_arithmetics_counter_type;

HPX_REGISTER_DERIVED_COMPONENT_FACTORY(
    variance_arithmetics_counter_type, variance_arithmetics_counter,
    "base_performance_counter", hpx::components::factory_enabled)
HPX_DEFINE_GET_COMPONENT_TYPE(variance_arithmetics_counter_type::wrapped_type)

///////////////////////////////////////////////////////////////////////////////
// /arithmetic/median
typedef hpx::components::component<
    hpx::performance_counters::server::arithmetics_counter_extended<
        boost::accumulators::tag::median>
> median_arithmetics_counter_type;

HPX_REGISTER_DERIVED_COMPONENT_FACTORY(
    median_arithmetics_counter_type, median_arithmetics_counter,
    "base_performance_counter", hpx::components::factory_enabled)
HPX_DEFINE_GET_COMPONENT_TYPE(median_arithmetics_counter_type::wrapped_type)

///////////////////////////////////////////////////////////////////////////////
// /arithmetic/min
typedef hpx::components::component<
    hpx::performance_counters::server::arithmetics_counter_extended<
        boost::accumulators::tag::min>
> min_arithmetics_counter_type;

HPX_REGISTER_DERIVED_COMPONENT_FACTORY(
    min_arithmetics_counter_type, min_arithmetics_counter,
    "base_performance_counter", hpx::components::factory_enabled)
HPX_DEFINE_GET_COMPONENT_TYPE(min_arithmetics_counter_type::wrapped_type)

///////////////////////////////////////////////////////////////////////////////
// /arithmetic/max
typedef hpx::components::component<
    hpx::performance_counters::server::arithmetics_counter_extended<
        boost::accumulators::tag::max>
> max_arithmetics_counter_type;

HPX_REGISTER_DERIVED_COMPONENT_FACTORY(
    max_arithmetics_counter_type, max_arithmetics_counter,
    "base_performance_counter", hpx::components::factory_enabled)
HPX_DEFINE_GET_COMPONENT_TYPE(max_arithmetics_counter_type::wrapped_type)

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace performance_counters { namespace detail
{
    /// Creation function for aggregating performance counters to be registered
    /// with the counter types.
    naming::gid_type arithmetics_counter_extended_creator(
        counter_info const& info, error_code& ec)
    {
        switch (info.type_) {
        case counter_aggregating:
            {
                counter_path_elements paths;
                get_counter_path_elements(info.fullname_, paths, ec);
                if (ec) return naming::invalid_gid;

                if (!paths.parameters_.empty()) {
                    // try to interpret the additional parameter as a list of
                    // two performance counter names
                    std::vector<std::string> names;
                    boost::split(names, paths.parameters_, boost::is_any_of(","));

                    if (names.empty())
                    {
                        HPX_THROWS_IF(ec, bad_parameter,
                            "arithmetics_counter_extended_creator",
                            "the parameter specification for an arithmetic counter "
                            "has to expand to at least one counter name: " +
                            paths.parameters_);
                        return naming::invalid_gid;
                    }

                    for (std::string const& name : names)
                    {
                        counter_path_elements paths;
                        if (status_valid_data != get_counter_path_elements(
                                name, paths, ec) || ec)
                        {
                            HPX_THROWS_IF(ec, bad_parameter,
                                "arithmetics_counter_extended_creator",
                                "the given (expanded) counter name is not "
                                "a validly formed performance counter name: " +
                                    name);
                            return naming::invalid_gid;
                        }
                    }

                    return create_arithmetics_counter_extended(info, names, ec);
                }
                else {
                    HPX_THROWS_IF(ec, bad_parameter,
                        "arithmetics_counter_extended_creator",
                        "the parameter specification for an arithmetic counter "
                        "has to be a comma separated list of performance "
                        "counter names, none is given: " +
                            remove_counter_prefix(info.fullname_));
                }
            }
            break;

        default:
            HPX_THROWS_IF(ec, bad_parameter,
                "arithmetics_counter_extended_creator",
                "invalid counter type requested");
            break;
        }
        return naming::invalid_gid;
    }
}}}

