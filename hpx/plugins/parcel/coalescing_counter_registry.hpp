//  Copyright (c) 2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_RUNTIME_COALESCING_COUNTER_REGISTRY_MAR_16_2016_0821PM)
#define HPX_RUNTIME_COALESCING_COUNTER_REGISTRY_MAR_16_2016_0821PM

#include <hpx/config.hpp>

#if defined(HPX_HAVE_PARCEL_COALESCING)

#include <hpx/performance_counters/counters.hpp>
#include <hpx/util/move.hpp>
#include <hpx/util/tuple.hpp>
#include <hpx/util/jenkins_hash.hpp>
#include <hpx/util/static.hpp>

#include <boost/unordered_map.hpp>
#include <boost/cstdint.hpp>

#include <string>

#include <hpx/config/warnings_prefix.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace plugins { namespace parcel
{
    ///////////////////////////////////////////////////////////////////////////
    class coalescing_counter_registry
    {
        HPX_MOVABLE_BUT_NOT_COPYABLE(coalescing_counter_registry);

    public:
        coalescing_counter_registry() {}

        typedef boost::int64_t (*get_counter_type)(bool);
        typedef util::tuple<
                get_counter_type, get_counter_type, get_counter_type
            > counter_types;
        typedef boost::unordered_map<
                std::string, counter_types, hpx::util::jenkins_hash
            > map_type;

        static coalescing_counter_registry& instance();

        void register_message_handler(std::string const& name);

        void register_message_handler(std::string const& name,
            get_counter_type num_parcels, get_counter_type num_messages,
            get_counter_type time_between_parcels);

        get_counter_type get_num_parcels_counter(std::string const& name) const;
        get_counter_type get_num_messages_counter(std::string const& name) const;
        get_counter_type get_average_time_between_parcels_counter(
            std::string const& name) const;

        bool counter_discoverer(
            performance_counters::counter_info const& info,
            performance_counters::counter_path_elements& p,
            performance_counters::discover_counter_func const& f,
            performance_counters::discover_counters_mode mode, error_code& ec);

    private:
        struct tag {};

        friend struct hpx::util::static_<
                coalescing_counter_registry, tag
            >;

        map_type map_;
    };

    ///////////////////////////////////////////////////////////////////////////
    void register_coalescing_counters(char const* action_name);

    template <typename Action>
    struct register_coalescing_for_action
    {
        register_coalescing_for_action()
        {
            register_coalescing_counters(
                hpx::actions::detail::get_action_name<Action>());
        }
        static register_coalescing_for_action instance_;
    };

    template <typename Action>
    register_coalescing_for_action<Action>
        register_coalescing_for_action<Action>::instance_;
}}}

#include <hpx/config/warnings_suffix.hpp>

///////////////////////////////////////////////////////////////////////////////
#define HPX_REGISTER_COALESCING_COUNTERS(Action)                              \
    namespace hpx { namespace plugins { namespace parcel                      \
    {                                                                         \
        template register_coalescing_for_action<Action>                       \
            register_coalescing_for_action<Action>::instance_;                \
    }}}                                                                       \
/**/

#endif

#endif
