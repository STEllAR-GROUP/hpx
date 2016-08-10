//  Copyright (c) 2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_RUNTIME_COALESCING_COUNTER_REGISTRY_MAR_16_2016_0821PM)
#define HPX_RUNTIME_COALESCING_COUNTER_REGISTRY_MAR_16_2016_0821PM

#include <hpx/config.hpp>

#if defined(HPX_HAVE_PARCEL_COALESCING)

#include <hpx/performance_counters/counters.hpp>
#include <hpx/util/jenkins_hash.hpp>
#include <hpx/util/function.hpp>
#include <hpx/util/static.hpp>

#include <boost/cstdint.hpp>

#include <string>
#include <unordered_map>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace plugins { namespace parcel
{
    ///////////////////////////////////////////////////////////////////////////
    class coalescing_counter_registry
    {
        HPX_MOVABLE_ONLY(coalescing_counter_registry);

    public:
        coalescing_counter_registry() {}

        typedef util::function_nonser<boost::int64_t(bool)>
            get_counter_type;
        typedef util::function_nonser<std::vector<boost::int64_t>(bool)>
            get_counter_values_type;
        typedef util::function_nonser<
                void(boost::int64_t, boost::int64_t, boost::int64_t,
                    get_counter_values_type&)
            > get_counter_values_creator_type;

        struct counter_functions
        {
            get_counter_type num_parcels;
            get_counter_type num_messages;
            get_counter_type num_parcels_per_message;
            get_counter_type average_time_between_parcels;
            get_counter_values_creator_type time_between_parcels_histogram_creator;
            boost::int64_t min_boundary, max_boundary, num_buckets;
        };

        typedef std::unordered_map<
                std::string, counter_functions, hpx::util::jenkins_hash
            > map_type;

        static coalescing_counter_registry& instance();

        void register_action(std::string const& name);

        void register_action(std::string const& name,
            get_counter_type num_parcels, get_counter_type num_messages,
            get_counter_type time_between_parcels,
            get_counter_type average_time_between_parcels,
            get_counter_values_creator_type time_between_parcels_histogram_creator);

        get_counter_type get_parcels_counter(std::string const& name) const;
        get_counter_type get_messages_counter(std::string const& name) const;
        get_counter_type get_parcels_per_message_counter(
            std::string const& name) const;
        get_counter_type get_average_time_between_parcels_counter(
            std::string const& name) const;
        get_counter_values_type get_time_between_parcels_histogram_counter(
            std::string const& name, boost::int64_t min_boundary,
            boost::int64_t max_boundary, boost::int64_t num_buckets);

        bool counter_discoverer(
            performance_counters::counter_info const& info,
            performance_counters::counter_path_elements& p,
            performance_counters::discover_counter_func const& f,
            performance_counters::discover_counters_mode mode, error_code& ec);

        static std::vector<boost::int64_t> empty_histogram(bool)
        {
            std::vector<boost::int64_t> result = { 0, 0, 1, 0 };
            return result;
        }

    private:
        struct tag {};

        friend struct hpx::util::static_<
                coalescing_counter_registry, tag
            >;

        map_type map_;
    };
}}}

#endif
#endif
