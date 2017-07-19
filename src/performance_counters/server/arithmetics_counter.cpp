//  Copyright (c) 2007-2017 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// make inspect happy: hpxinspect:nodeprecatedname:boost::is_any_of

#include <hpx/config.hpp>
#include <hpx/runtime/runtime_fwd.hpp>
#include <hpx/runtime/components/derived_component_factory.hpp>
#include <hpx/runtime/actions/continuation.hpp>
#include <hpx/runtime/agas/interface.hpp>
#include <hpx/runtime/launch_policy.hpp>
#include <hpx/util/high_resolution_clock.hpp>
#include <hpx/performance_counters/counters.hpp>
#include <hpx/performance_counters/counter_creators.hpp>
#include <hpx/performance_counters/stubs/performance_counter.hpp>
#include <hpx/performance_counters/server/arithmetics_counter.hpp>

#include <boost/algorithm/string.hpp>

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
    namespace detail
    {
        template <typename Operation>
        struct init_value;

        template <>
        struct init_value<std::plus<double> >
        {
            static double call() { return 0.0; }
        };

        template <>
        struct init_value<std::minus<double> >
        {
            static double call() { return 0.0; }
        };

        template <>
        struct init_value<std::multiplies<double> >
        {
            static double call() { return 1.0; }
        };

        template <>
        struct init_value<std::divides<double> >
        {
            static double call() { return 1.0; }
        };
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename Operation>
    arithmetics_counter<Operation>::arithmetics_counter(
            counter_info const& info,
            std::vector<std::string> const& base_counter_names)
      : base_type_holder(info),
        counters_(base_counter_names)
    {
        if (info.type_ != counter_aggregating) {
            HPX_THROW_EXCEPTION(bad_parameter,
                "arithmetics_counter<Operation>::arithmetics_counter",
                "unexpected counter type specified");
        }

        counter_path_elements paths;
        get_counter_path_elements(info.fullname_, paths);

        if (paths.countername_ == "divide")
        {
            if (counters_.size() < 2)
            {
                HPX_THROW_EXCEPTION(bad_parameter,
                    "arithmetics_counter<Operation>::arithmetics_counter",
                    "the parameter specification for an arithmetic counter "
                    "'/arithmetics/divide' has to expand to more than one "
                    "counter name: " + paths.parameters_);
            }
        }
    }

    template <typename Operation>
    hpx::performance_counters::counter_value
        arithmetics_counter<Operation>::get_counter_value(bool reset)
    {
        std::vector<counter_value> base_values =
            counters_.get_counter_values(hpx::launch::sync);

        // apply arithmetic operation
        double value = detail::init_value<Operation>::call();
        for (counter_value const& base_value : base_values)
        {
            value = Operation()(value, base_value.get_value<double>());
        }

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

    template <typename Operation>
    bool arithmetics_counter<Operation>::start()
    {
        return counters_.start(hpx::launch::sync);
    }

    template <typename Operation>
    bool arithmetics_counter<Operation>::stop()
    {
        return counters_.stop(hpx::launch::sync);
    }

    template <typename Operation>
    void arithmetics_counter<Operation>::reset_counter_value()
    {
        counters_.reset(hpx::launch::sync);
    }
}}}

///////////////////////////////////////////////////////////////////////////////
template class HPX_EXPORT hpx::performance_counters::server::arithmetics_counter<
    std::plus<double> >;
template class HPX_EXPORT hpx::performance_counters::server::arithmetics_counter<
    std::minus<double> >;
template class HPX_EXPORT hpx::performance_counters::server::arithmetics_counter<
    std::multiplies<double> >;
template class HPX_EXPORT hpx::performance_counters::server::arithmetics_counter<
    std::divides<double> >;

///////////////////////////////////////////////////////////////////////////////
// Addition
typedef hpx::components::component<
    hpx::performance_counters::server::arithmetics_counter<std::plus<double> >
> adding_counter_type;

HPX_REGISTER_DERIVED_COMPONENT_FACTORY(
    adding_counter_type, adding_counter,
    "base_performance_counter", hpx::components::factory_enabled)
HPX_DEFINE_GET_COMPONENT_TYPE(adding_counter_type::wrapped_type)

///////////////////////////////////////////////////////////////////////////////
// Subtraction
typedef hpx::components::component<
    hpx::performance_counters::server::arithmetics_counter<std::minus<double> >
> subtracting_counter_type;

HPX_REGISTER_DERIVED_COMPONENT_FACTORY(
    subtracting_counter_type, subtracting_counter,
    "base_performance_counter", hpx::components::factory_enabled)
HPX_DEFINE_GET_COMPONENT_TYPE(subtracting_counter_type::wrapped_type)

///////////////////////////////////////////////////////////////////////////////
// Multiply
typedef hpx::components::component<
    hpx::performance_counters::server::arithmetics_counter<std::multiplies<double> >
> multiplying_counter_type;

HPX_REGISTER_DERIVED_COMPONENT_FACTORY(
    multiplying_counter_type, multiplying_counter,
    "base_performance_counter", hpx::components::factory_enabled)
HPX_DEFINE_GET_COMPONENT_TYPE(multiplying_counter_type::wrapped_type)

///////////////////////////////////////////////////////////////////////////////
// Division
typedef hpx::components::component<
    hpx::performance_counters::server::arithmetics_counter<std::divides<double> >
> dividing_counter_type;

HPX_REGISTER_DERIVED_COMPONENT_FACTORY(
    dividing_counter_type, dividing_counter,
    "base_performance_counter", hpx::components::factory_enabled)
HPX_DEFINE_GET_COMPONENT_TYPE(dividing_counter_type::wrapped_type)

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace performance_counters { namespace detail
{
    /// Creation function for aggregating performance counters to be registered
    /// with the counter types.
    naming::gid_type arithmetics_counter_creator(counter_info const& info,
        error_code& ec)
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
                            "arithmetics_counter_creator",
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
                                "arithmetics_counter_creator",
                                "the given (expanded) counter name is not "
                                "a validly formed performance counter name: " +
                                    name);
                            return naming::invalid_gid;
                        }
                    }

                    return create_arithmetics_counter(info, names, ec);
                }
                else {
                    HPX_THROWS_IF(ec, bad_parameter,
                        "arithmetics_counter_creator",
                        "the parameter specification for an arithmetic counter "
                        "has to be a comma separated list of performance "
                        "counter names, none is given: " +
                            remove_counter_prefix(info.fullname_));
                }
            }
            break;

        default:
            HPX_THROWS_IF(ec, bad_parameter, "arithmetics_counter_creator",
                "invalid counter type requested");
            break;
        }
        return naming::invalid_gid;
    }
}}}

