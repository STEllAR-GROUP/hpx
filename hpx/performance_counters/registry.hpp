//  Copyright (c) 2007-2011 Hartmut Kaiser
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
    class registry
    {
    private:
        typedef std::map<std::string, counter_info> counter_type_map_type;

    public:
        registry(naming::resolver_client& agas_client);

        /// \brief Add a new performance counter type to the (local) registry
        counter_status add_counter_type(counter_info const& info,
            error_code& ec = throws);

        /// \brief Remove an existing counter type from the (local) registry
        ///
        /// \note This doesn't remove existing counters of this type, it just
        ///       inhibits defining new counters using this type.
        counter_status remove_counter_type(counter_info const& info,
            error_code& ec = throws);

        /// \brief Create a new performance counter instance of type
        ///        raw_counter based on given counter value
        counter_status create_raw_counter(counter_info const& info,
            boost::int64_t* countervalue, naming::id_type& id,
            error_code& ec = throws);

        /// \brief Create a new performance counter instance of type
        ///        raw_counter based on given function returning the counter
        ///        value
        counter_status create_raw_counter(counter_info const& info,
            HPX_STD_FUNCTION<boost::int64_t()> f, naming::id_type& id,
            error_code& ec = throws);

        /// \brief Create a new performance counter instance based on given
        ///        counter info
        counter_status create_counter(counter_info const& info,
            naming::id_type& id, error_code& ec = throws);

        /// \brief Create a new performance counter instance of type
        ///        counter_average_count based on given base counter name and
        ///        given base time interval (milliseconds)
        counter_status create_average_count_counter(counter_info const& info,
            std::string const& base_counter_name, std::size_t base_time_interval,
            naming::id_type& id, error_code& ec = throws);

        /// \brief Add an existing performance counter instance to the registry
        counter_status add_counter(naming::id_type const& id,
            counter_info const& info, error_code& ec = throws);

        /// \brief remove the existing performance counter from the registry
        counter_status remove_counter(counter_info const& info,
            naming::id_type const& id, error_code& ec = throws);

    private:
        naming::resolver_client& agas_client_;
        counter_type_map_type countertypes_;
    };
}}

#endif

