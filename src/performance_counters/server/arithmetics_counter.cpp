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
#include <hpx/performance_counters/server/arithmetics_counter.hpp>

#include <boost/algorithm/string.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace performance_counters { namespace server
{
    ///////////////////////////////////////////////////////////////////////////
    template <typename Operation>
    arithmetics_counter<Operation>::arithmetics_counter(
            counter_info const& info, std::string const& base_counter_name1,
            std::string const& base_counter_name2)
      : base_type_holder(info),
        base_counter_name1_(ensure_counter_prefix(base_counter_name1)),
        base_counter_name2_(ensure_counter_prefix(base_counter_name2))
    {
        if (info.type_ != counter_aggregating) {
            HPX_THROW_EXCEPTION(bad_parameter,
                "arithmetics_counter<Operation>::arithmetics_counter",
                "unexpected counter type specified");
        }
    }

    template <typename Operation>
    hpx::performance_counters::counter_value
        arithmetics_counter<Operation>::get_counter_value(bool reset)
    {
        counter_value base_value1, base_value2;

        {
            // lock here to avoid checking out multiple reference counted GIDs
            // from AGAS
            mutex_type::scoped_lock l(mtx_);

            // gather current base values
            if (!evaluate_base_counter(base_counter_id1_, base_counter_name1_, base_value1) ||
                !evaluate_base_counter(base_counter_id2_, base_counter_name2_, base_value2))
            {
                return false;
            }
        }

//         if (base_value1.scaling_ != base_value2.scaling_ ||
//             base_value1.scale_inverse_ != base_value2.scale_inverse_)
//         {
//             // not supported right now
//             HPX_THROW_EXCEPTION(not_implemented,
//                 "arithmetics_counter<Operation>::evaluate",
//                 "base counters should expose same scaling");
//             return false;
//         }

        // apply arithmetic operation
        double value = Operation()(base_value1.get_value<double>(),
            base_value2.get_value<double>());

        if (base_value1.scale_inverse_) {
            base_value1.value_ = static_cast<boost::int64_t>(value * base_value1.scaling_);
        }
        else {
            base_value1.value_ = static_cast<boost::int64_t>(value / base_value1.scaling_);
        }
        return base_value1;
    }

    template <typename Operation>
    bool arithmetics_counter<Operation>::ensure_base_counter(
        naming::id_type& base_counter_id, std::string const& base_counter_name)
    {
        if (!base_counter_id) {
            // get or create the base counter
            error_code ec(lightweight);
            base_counter_id = get_counter(base_counter_name, ec);
            if (HPX_UNLIKELY(ec || !base_counter_id))
            {
                // base counter could not be retrieved
                HPX_THROW_EXCEPTION(bad_parameter,
                    "arithmetics_counter<Operation>::evaluate_base_counter",
                    boost::str(boost::format(
                        "could not get or create performance counter: '%s'") %
                            base_counter_name));
                return false;
            }
        }
        return true;
    }

    template <typename Operation>
    bool arithmetics_counter<Operation>::evaluate_base_counter(
        naming::id_type& base_counter_id, std::string const& name,
        counter_value& value)
    {
        // query the actual value
        if (!base_counter_id && !ensure_base_counter(base_counter_id, name))
            return false;

        value = stubs::performance_counter::get_value(base_counter_id);
        return true;
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
typedef hpx::components::managed_component<
    hpx::performance_counters::server::arithmetics_counter<std::plus<double> >
> adding_counter_type;

HPX_REGISTER_DERIVED_COMPONENT_FACTORY(
    adding_counter_type, adding_counter,
    "base_performance_counter", hpx::components::factory_enabled)
HPX_DEFINE_GET_COMPONENT_TYPE(adding_counter_type::wrapped_type)

///////////////////////////////////////////////////////////////////////////////
// Subtraction
typedef hpx::components::managed_component<
    hpx::performance_counters::server::arithmetics_counter<std::minus<double> >
> subtracting_counter_type;

HPX_REGISTER_DERIVED_COMPONENT_FACTORY(
    subtracting_counter_type, subtracting_counter,
    "base_performance_counter", hpx::components::factory_enabled)
HPX_DEFINE_GET_COMPONENT_TYPE(subtracting_counter_type::wrapped_type)

///////////////////////////////////////////////////////////////////////////////
// Multiply
typedef hpx::components::managed_component<
    hpx::performance_counters::server::arithmetics_counter<std::multiplies<double> >
> multiplying_counter_type;

HPX_REGISTER_DERIVED_COMPONENT_FACTORY(
    multiplying_counter_type, multiplying_counter,
    "base_performance_counter", hpx::components::factory_enabled)
HPX_DEFINE_GET_COMPONENT_TYPE(multiplying_counter_type::wrapped_type)

///////////////////////////////////////////////////////////////////////////////
// Division
typedef hpx::components::managed_component<
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

                std::vector<boost::int64_t> parameters;
                if (!paths.parameters_.empty()) {
                    // try to interpret the additional parameter as a list of
                    // two performance counter names
                    std::vector<std::string> names;
                    boost::split(names, paths.parameters_, boost::is_any_of(","));

                    counter_path_elements paths1, paths2;
                    if (names.size() != 2)
                    {
                        HPX_THROWS_IF(ec, bad_parameter,
                            "arithmetics_counter_creator",
                            "the parameter specification for an arithmetic counter "
                            "has to be a comma separated list of two performance "
                            "counter names: " + paths.parameters_);
                        return naming::invalid_gid;
                    }

                    if (status_valid_data != get_counter_path_elements(
                            names[0], paths1, ec) || ec)
                    {
                        HPX_THROWS_IF(ec, bad_parameter,
                            "arithmetics_counter_creator",
                            "the first parameter is not a validly formed "
                            "performance counter name: " + names[0]);
                        return naming::invalid_gid;
                    }

                    if (status_valid_data != get_counter_path_elements(
                            names[1], paths2, ec) || ec)
                    {
                        HPX_THROWS_IF(ec, bad_parameter,
                            "arithmetics_counter_creator",
                            "the second parameter is not a validly formed "
                            "performance counter name: " + names[1]);
                        return naming::invalid_gid;
                    }

                    return create_arithmetics_counter(info, names[0], names[1], ec);
                }
                else {
                    HPX_THROWS_IF(ec, bad_parameter,
                        "arithmetics_counter_creator",
                        "the parameter specification for an arithmetic counter "
                        "has to be a comma separated list of two performance "
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

