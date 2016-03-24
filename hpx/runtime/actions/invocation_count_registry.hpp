//  Copyright (c) 2015 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_ACTIONS_INVOCATION_COUNT_REGISTRY_SEP_25_2015_0727AM)
#define HPX_ACTIONS_INVOCATION_COUNT_REGISTRY_SEP_25_2015_0727AM

#include <hpx/config.hpp>
#include <hpx/performance_counters/counters.hpp>

#include <hpx/util/jenkins_hash.hpp>
#include <hpx/util/static.hpp>

#include <boost/preprocessor/cat.hpp>
#include <boost/noncopyable.hpp>
#include <boost/atomic.hpp>

#include <unordered_map>

#include <hpx/config/warnings_prefix.hpp>

namespace hpx { namespace actions { namespace detail
{
    class HPX_EXPORT invocation_count_registry : boost::noncopyable
    {
    public:
        typedef boost::int64_t (*get_invocation_count_type)(bool);
        typedef std::unordered_map<
                std::string, get_invocation_count_type, hpx::util::jenkins_hash
            > map_type;

        static invocation_count_registry& local_instance();
        static invocation_count_registry& remote_instance();

        void register_class(std::string const& name, get_invocation_count_type fun);

        get_invocation_count_type
            get_invocation_counter(std::string const& name) const;

        bool counter_discoverer(
            performance_counters::counter_info const& info,
            performance_counters::counter_path_elements& p,
            performance_counters::discover_counter_func const& f,
            performance_counters::discover_counters_mode mode, error_code& ec);

    private:
        struct local_tag {};
        struct remote_tag {};

        friend struct hpx::util::static_<invocation_count_registry, local_tag>;
        friend struct hpx::util::static_<invocation_count_registry, remote_tag>;

        map_type map_;
    };

    template <typename Action>
    void register_local_action_invocation_count(
        invocation_count_registry& registry);

    template <typename Action>
    void register_remote_action_invocation_count(
        invocation_count_registry& registry);

    template <typename Action>
    struct register_action_invocation_count
    {
        register_action_invocation_count()
        {
            register_local_action_invocation_count<Action>(
                invocation_count_registry::local_instance());

            register_remote_action_invocation_count<Action>(
                invocation_count_registry::remote_instance());
        }

        static register_action_invocation_count instance;
    };

    template <typename Action>
    register_action_invocation_count<Action>
        register_action_invocation_count<Action>::instance;
}}}

#define HPX_REGISTER_ACTION_INVOCATION_COUNT(Action)                          \
    namespace hpx { namespace actions { namespace detail                      \
    {                                                                         \
        template register_action_invocation_count<Action>                     \
            register_action_invocation_count<Action>::instance;               \
    }}}                                                                       \
/**/

#include <hpx/config/warnings_suffix.hpp>

#endif
