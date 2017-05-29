//  Copyright (c) 2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PARCELSET_ACTION_DATA_COUNTER_REGISTRY_AUG_04_2016_0729PM)
#define HPX_PARCELSET_ACTION_DATA_COUNTER_REGISTRY_AUG_04_2016_0729PM

#include <hpx/config.hpp>

#if defined(HPX_HAVE_PARCELPORT_ACTION_COUNTERS)
#include <hpx/performance_counters/counters.hpp>

#include <hpx/util/jenkins_hash.hpp>
#include <hpx/util/function.hpp>
#include <hpx/util/static.hpp>

#include <boost/atomic.hpp>
#include <boost/preprocessor/cat.hpp>

#include <cstdint>
#include <string>
#include <unordered_set>
#include <vector>
#include <utility>

#include <hpx/config/warnings_prefix.hpp>

namespace hpx { namespace parcelset { namespace detail
{
    class HPX_EXPORT per_action_data_counter_registry
    {
    public:
        HPX_NON_COPYABLE(per_action_data_counter_registry);

    public:
        typedef util::function_nonser<std::int64_t(bool)> counter_function_type;

        enum per_action_counter_type
        {
            num_parcels = 0,
            num_messages,
            total_time,
            total_serialization_time,
            total_bytes,
            total_raw_bytes,
            total_buffer_allocate_time
        };

        typedef std::unordered_set<std::string, hpx::util::jenkins_hash> map_type;

        per_action_data_counter_registry() {}

        static per_action_data_counter_registry& instance();

        void register_class(std::string action);

        counter_function_type get_counter(std::string const& action,
            hpx::util::function_nonser<
                std::int64_t(std::string const&, bool)
            > const& f) const;

        bool counter_discoverer(
            performance_counters::counter_info const& info,
            performance_counters::counter_path_elements& p,
            performance_counters::discover_counter_func const& f,
            performance_counters::discover_counters_mode mode,
            error_code& ec);

    private:
        struct tag {};
        friend struct hpx::util::static_<per_action_data_counter_registry, tag>;

        map_type map_;
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename Action>
    void register_per_action_data_counter_types(
        per_action_data_counter_registry& registry);

    template <typename Action>
    struct register_per_action_data_counters
    {
        register_per_action_data_counters()
        {
            register_per_action_data_counter_types<Action>(
                per_action_data_counter_registry::instance());
        }

        static register_per_action_data_counters instance;
    };

    template <typename Action>
    register_per_action_data_counters<Action>
        register_per_action_data_counters<Action>::instance;
}}}

#define HPX_REGISTER_PER_ACTION_DATA_COUNTER_TYPES(Action)                    \
    namespace hpx { namespace parcelset { namespace detail                    \
    {                                                                         \
        template register_per_action_data_counters< Action>                   \
            register_per_action_data_counters< Action>::instance;             \
    }}}                                                                       \
/**/

#include <hpx/config/warnings_suffix.hpp>

#else

#define HPX_REGISTER_PER_ACTION_DATA_COUNTER_TYPES(Action)

#endif

#endif
