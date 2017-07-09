//  Copyright (c) 2007-2013 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// make inspect happy: hpxinspect:nodeprecatedname:boost::is_any_of

#include <hpx/config.hpp>
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
        base_counter_names_(base_counter_names),
        invocation_count_(0)
    {
        if (info.type_ != counter_aggregating) {
            HPX_THROW_EXCEPTION(bad_parameter,
                "arithmetics_counter<Operation>::arithmetics_counter",
                "unexpected counter type specified");
        }
        base_counter_ids_.resize(base_counter_names_.size());
    }

    template <typename Operation>
    hpx::performance_counters::counter_value
        arithmetics_counter<Operation>::get_counter_value(bool reset)
    {
        std::vector<counter_value> base_values;
        base_values.reserve(base_counter_names_.size());

        // lock here to avoid checking out multiple reference counted GIDs
        // from AGAS
        {
            std::unique_lock<mutex_type> l(mtx_);

            for (std::size_t i = 0; i != base_counter_names_.size(); ++i)
            {
                // gather current base values
                counter_value value;
                if (!evaluate_base_counter(base_counter_ids_[i],
                    base_counter_names_[i], value, l))
                {
                    return value;
                }

                base_values.push_back(value);
            }

            // adjust local invocation count
            base_values[0].count_ = ++invocation_count_;
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
        return base_values[0];
    }

    template <typename Operation>
    bool arithmetics_counter<Operation>::start()
    {
        std::unique_lock<mutex_type> l(mtx_);
        for (std::size_t i = 0; i != base_counter_names_.size(); ++i)
        {
            if (!base_counter_ids_[i] &&
                !ensure_base_counter(base_counter_ids_[i], base_counter_names_[i], l))
            {
                HPX_THROW_EXCEPTION(bad_parameter,
                    "arithmetics_counter<Operation>::start",
                    boost::str(boost::format(
                        "could not get or create performance counter: '%s'") %
                            base_counter_names_[i]));
                return false;
            }

            {
                util::unlock_guard<std::unique_lock<mutex_type> > ul(l);
                using performance_counters::stubs::performance_counter;
                if (!performance_counter::start(launch::sync, base_counter_ids_[i]))
                {
                    HPX_THROW_EXCEPTION(bad_parameter,
                        "arithmetics_counter<Operation>::stop",
                        boost::str(boost::format(
                            "could not start performance counter: '%s'") %
                                base_counter_names_[i]));
                    return false;
                }
            }
        }
        return true;
    }

    template <typename Operation>
    bool arithmetics_counter<Operation>::stop()
    {
        std::unique_lock<mutex_type> l(mtx_);
        for (std::size_t i = 0; i != base_counter_names_.size(); ++i)
        {
            if (!base_counter_ids_[i] &&
                !ensure_base_counter(base_counter_ids_[i], base_counter_names_[i], l))
            {
                HPX_THROW_EXCEPTION(bad_parameter,
                    "arithmetics_counter<Operation>::stop",
                    boost::str(boost::format(
                        "could not get or create performance counter: '%s'") %
                            base_counter_names_[i]));
                return false;
            }

            {
                util::unlock_guard<std::unique_lock<mutex_type> > ul(l);
                using performance_counters::stubs::performance_counter;
                if (!performance_counter::stop(launch::sync, base_counter_ids_[i]))
                {
                    HPX_THROW_EXCEPTION(bad_parameter,
                        "arithmetics_counter<Operation>::stop",
                        boost::str(boost::format(
                            "could not stop performance counter: '%s'") %
                                base_counter_names_[i]));
                    return false;
                }
            }
        }
        return true;
    }

    template <typename Operation>
    void arithmetics_counter<Operation>::reset_counter_value()
    {
        std::unique_lock<mutex_type> l(mtx_);
        for (std::size_t i = 0; i != base_counter_names_.size(); ++i)
        {
            if (!base_counter_ids_[i] &&
                !ensure_base_counter(base_counter_ids_[i], base_counter_names_[i], l))
            {
                HPX_THROW_EXCEPTION(bad_parameter,
                    "arithmetics_counter<Operation>::reset_counter_value",
                    boost::str(boost::format(
                        "could not get or create performance counter: '%s'") %
                            base_counter_names_[i]));
                return;
            }

            using performance_counters::stubs::performance_counter;
            performance_counter::reset(launch::sync, base_counter_ids_[i]);
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename Operation>
    bool arithmetics_counter<Operation>::ensure_base_counter(
        naming::id_type& base_counter_id, std::string const& base_counter_name,
        std::unique_lock<mutex_type>& l)
    {
        if (!base_counter_id) {
            // get or create the base counter
            error_code ec(lightweight);
            hpx::id_type counter_id;
            {
                util::unlock_guard<std::unique_lock<mutex_type> > ul(l);
                counter_id = get_counter(base_counter_name, ec);
            }
            // Since we needed to unlock to retrieve the counter id, check if
            // no other thread came first
            if (!base_counter_id)
            {
                base_counter_id = counter_id;
            }
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
        counter_value& value, std::unique_lock<mutex_type>& l)
    {
        // query the actual value
        if (!base_counter_id && !ensure_base_counter(base_counter_id, name, l))
            return false;

        hpx::id_type counter_id = base_counter_id;
        {
            util::unlock_guard<std::unique_lock<mutex_type> > ul(l);
            value = stubs::performance_counter::get_value(launch::sync, counter_id);
        }
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
    void expand_counter_name_wildcards(std::vector<std::string>& names, error_code& ec)
    {
        std::vector<counter_info> counters;
        for (std::string const& name : names)
        {
            discover_counter_type(ensure_counter_prefix(name), counters,
                discover_counters_full, ec);
            if (ec) return;
        }

        std::vector<std::string> result;
        result.reserve(counters.size());
        for (counter_info const& info : counters)
        {
            result.push_back(info.fullname_);
        }

        if (&ec != &throws)
            ec = make_success_code();

        std::swap(names, result);
    }

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

                    // expand all wildcards in given counter names
                    expand_counter_name_wildcards(names, ec);
                    if (ec) return naming::invalid_gid;

                    for (std::string const& name : names)
                    {
                        counter_path_elements paths;
                        if (status_valid_data != get_counter_path_elements(
                                name, paths, ec) || ec)
                        {
                            HPX_THROWS_IF(ec, bad_parameter,
                                "arithmetics_counter_creator",
                                "the given (expanded) counter name is not \
                                 a validly formed "
                                "performance counter name: " + name);
                            return naming::invalid_gid;
                        }
                    }

                    if (paths.countername_ == "divide")
                    {
                        if (names.size() < 2)
                        {
                            HPX_THROWS_IF(ec, bad_parameter,
                                "arithmetics_counter_creator",
                                "the parameter specification for an arithmetic counter "
                                "has to expand to more than one counter name: " +
                                paths.parameters_);
                            return naming::invalid_gid;
                        }
                    }
                    else if (names.empty())
                    {
                        HPX_THROWS_IF(ec, bad_parameter,
                            "arithmetics_counter_creator",
                            "the parameter specification for an arithmetic counter "
                            "has to expand to at least one counter name: " +
                            paths.parameters_);
                        return naming::invalid_gid;
                    }

                    return create_arithmetics_counter(info, names, ec);
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

