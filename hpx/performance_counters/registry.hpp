//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PERFORMANCE_COUNTERS_REGISTRY_MAR_01_2009_0424PM)
#define HPX_PERFORMANCE_COUNTERS_REGISTRY_MAR_01_2009_0424PM

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/naming/name.hpp>
#include <hpx/runtime/naming/resolver_client.hpp>
#include <hpx/performance_counters/counters.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace performance_counters
{
    ///////////////////////////////////////////////////////////////////////////
    class registry
    {
    private:
        struct counter_data
        {
            counter_data(counter_info const& info,
                    HPX_STD_FUNCTION<create_counter_func> const& create_counter,
                    HPX_STD_FUNCTION<discover_counters_func> const& discover_counters)
              : info_(info), create_counter_(create_counter),
                discover_counters_(discover_counters)
            {}

            counter_info info_;
            HPX_STD_FUNCTION<create_counter_func> create_counter_;
            HPX_STD_FUNCTION<discover_counters_func> discover_counters_;
        };
        typedef std::map<std::string, counter_data> counter_type_map_type;

    public:
        registry(naming::resolver_client& agas_client);

        /// \brief Add a new performance counter type to the (local) registry
        counter_status add_counter_type(counter_info const& info,
            HPX_STD_FUNCTION<create_counter_func> const& create_counter,
            HPX_STD_FUNCTION<discover_counters_func> const& discover_counters,
            error_code& ec = throws);

        /// \brief Call the supplied function for all registered counter types.
        counter_status discover_counter_types(
            HPX_STD_FUNCTION<discover_counter_func> const& discover_counter,
            error_code& ec = throws);

        /// \brief Retrieve the counter creation function which is associated
        ///        with a given counter type.
        counter_status get_counter_create_function(counter_info const& info,
            HPX_STD_FUNCTION<create_counter_func>& create_counter,
            error_code& ec = throws) const;

        /// \brief Retrieve the counter discovery function which is associated
        ///        with a given counter type.
        counter_status get_counter_discovery_function(counter_info const& info,
            HPX_STD_FUNCTION<discover_counters_func>& func,
            error_code& ec) const;

        /// \brief Remove an existing counter type from the (local) registry
        ///
        /// \note This doesn't remove existing counters of this type, it just
        ///       inhibits defining new counters using this type.
        counter_status remove_counter_type(counter_info const& info,
            error_code& ec = throws);

        /// \brief Create a new performance counter instance of type
        ///        raw_counter based on given counter value
        counter_status create_raw_counter_value(counter_info const& info,
            boost::int64_t* countervalue, naming::gid_type& id,
            error_code& ec = throws);

        /// \brief Create a new performance counter instance of type
        ///        raw_counter based on given function returning the counter
        ///        value
        counter_status create_raw_counter(counter_info const& info,
            HPX_STD_FUNCTION<boost::int64_t()> const& f, naming::gid_type& id,
            error_code& ec = throws);

        /// \brief Create a new performance counter instance based on given
        ///        counter info
        counter_status create_counter(counter_info const& info,
            naming::gid_type& id, error_code& ec = throws);

        /// \brief Create a new aggregating performance counter instance based
        ///        on given base counter name and given base time interval
        ///        (milliseconds).
        counter_status create_aggregating_counter(counter_info const& info,
            std::string const& base_counter_name, boost::int64_t base_time_interval,
            naming::gid_type& id, error_code& ec = throws);

        /// \brief Add an existing performance counter instance to the registry
        counter_status add_counter(naming::id_type const& id,
            counter_info const& info, error_code& ec = throws);

        /// \brief remove the existing performance counter from the registry
        counter_status remove_counter(counter_info const& info,
            naming::id_type const& id, error_code& ec = throws);

        /// \brief Retrieve counter type information for given counter name
        counter_status get_counter_type(std::string const& name,
            counter_info& info, error_code& ec = throws);

    protected:
        counter_type_map_type::iterator
            locate_counter_type(std::string const& type_name);
        counter_type_map_type::const_iterator
            locate_counter_type(std::string const& type_name) const;

    private:
        naming::resolver_client& agas_client_;
        counter_type_map_type countertypes_;
    };
}}

#endif

