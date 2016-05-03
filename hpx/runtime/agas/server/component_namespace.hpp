////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Adelstein-Lelbach
//  Copyright (c) 2012-2013 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(HPX_A16135FC_AA32_444F_BB46_549AD456A661)
#define HPX_A16135FC_AA32_444F_BB46_549AD456A661

#include <hpx/config.hpp>
#include <hpx/runtime/agas_fwd.hpp>
#include <hpx/exception_fwd.hpp>
#include <hpx/runtime/actions/component_action.hpp>
#include <hpx/runtime/agas_fwd.hpp>
#include <hpx/runtime/components/component_type.hpp>
#include <hpx/runtime/components/server/fixed_component_base.hpp>
#include <hpx/util/function.hpp>
#include <hpx/lcos/local/spinlock.hpp>

#include <cstdint>
#include <map>
#include <set>
#include <string>
#include <vector>

#include <boost/atomic.hpp>
#include <boost/bimap.hpp>

namespace hpx { namespace agas
{

HPX_EXPORT naming::gid_type bootstrap_component_namespace_gid();
HPX_EXPORT naming::id_type bootstrap_component_namespace_id();

namespace server
{

// Base name used to register the component
char const* const component_namespace_service_name = "component/";

struct HPX_EXPORT component_namespace
  : components::fixed_component_base<component_namespace>
{
    // {{{ nested types
    typedef lcos::local::spinlock mutex_type;
    typedef components::fixed_component_base<component_namespace> base_type;

    typedef components::component_type component_id_type;

    typedef std::set<std::uint32_t> prefixes_type;

    typedef boost::bimap<std::string, component_id_type> component_id_table_type;

    typedef std::map<component_id_type, prefixes_type> factory_table_type;
    // }}}

  private:
    // REVIEW: Separate mutexes might reduce contention here. This has to be
    // investigated carefully.
    mutex_type mutex_;
    component_id_table_type component_ids_;
    factory_table_type factories_;
    component_id_type type_counter;
    std::string instance_name_;

    // data structure holding all counters for the omponent_namespace component
    struct counter_data
    {
    private:
        HPX_NON_COPYABLE(counter_data);

    public:
        typedef lcos::local::spinlock mutex_type;

        struct api_counter_data
        {
            api_counter_data()
                : count_(0)
                , time_(0)
            {}

            boost::atomic<std::int64_t> count_;
            boost::atomic<std::int64_t> time_;
        };

        counter_data()
        {}

    public:
        // access current counter values
        std::int64_t get_bind_prefix_count(bool);
        std::int64_t get_bind_name_count(bool);
        std::int64_t get_resolve_id_count(bool);
        std::int64_t get_unbind_name_count(bool);
        std::int64_t get_iterate_types_count(bool);
        std::int64_t get_component_type_name_count(bool);
        std::int64_t get_num_localities_count(bool);
        std::int64_t get_overall_count(bool);

        std::int64_t get_bind_prefix_time(bool);
        std::int64_t get_bind_name_time(bool);
        std::int64_t get_resolve_id_time(bool);
        std::int64_t get_unbind_name_time(bool);
        std::int64_t get_iterate_types_time(bool);
        std::int64_t get_component_type_name_time(bool);
        std::int64_t get_num_localities_time(bool);
        std::int64_t get_overall_time(bool);

        // increment counter values
        void increment_bind_prefix_count();
        void increment_bind_name_count();
        void increment_resolve_id_count();
        void increment_unbind_name_count();
        void increment_iterate_types_count();
        void increment_get_component_type_name_count();
        void increment_num_localities_count();

    private:
        friend struct component_namespace;

        api_counter_data bind_prefix_;          // component_ns_bind_prefix
        api_counter_data bind_name_;            // component_ns_bind_name
        api_counter_data resolve_id_;           // component_ns_resolve_id
        api_counter_data unbind_name_;          // component_ns_unbind_name
        api_counter_data iterate_types_;        // component_ns_iterate_types
        api_counter_data get_component_type_name_;
        // component_ns_get_component_type_name
        api_counter_data num_localities_;  // component_ns_num_localities
    };
    counter_data counter_data_;

  public:
    component_namespace()
      : base_type(HPX_AGAS_COMPONENT_NS_MSB, HPX_AGAS_COMPONENT_NS_LSB)
      , type_counter(components::component_first_dynamic)
    {}

    void finalize();

    // register all performance counter types exposed by this component
    static void register_counter_types(
        error_code& ec = throws
        );
    static void register_global_counter_types(
        error_code& ec = throws
        );

    void register_server_instance(
        char const* servicename
      , error_code& ec = throws
        );

    void unregister_server_instance(
        error_code& ec = throws
        );

    components::component_type bind_prefix(
        std::string const& key
      , boost::uint32_t prefix
        );

    components::component_type bind_name(
        std::string const& name
        );

    std::vector<boost::uint32_t> resolve_id(
        components::component_type key
        );

    bool unbind(
        std::string const& key
        );

    void iterate_types(
        iterate_types_function_type const& f
        );

    std::string get_component_type_name(
        components::component_type type
        );

    boost::uint32_t get_num_localities(
        components::component_type type
        );

    naming::gid_type statistics_counter(
        std::string const& name
        );

//     enum actions
//     { // {{{ action enum
//         // Actual actions
//         namespace_service                 = component_ns_service
//       , namespace_bulk_service            = component_ns_bulk_service
//
//         // Pseudo-actions
//       , namespace_bind_prefix             = component_ns_bind_prefix
//       , namespace_bind_name               = component_ns_bind_name
//       , namespace_resolve_id              = component_ns_resolve_id
//       , namespace_unbind_name             = component_ns_unbind_name
//       , namespace_iterate_types           = component_ns_iterate_types
//       , namespace_get_component_type_name = component_ns_get_component_type_name
//       , namespace_num_localities     = component_ns_num_localities
//       , namespace_statistics              = component_ns_statistics_counter
//     }; // }}}

    HPX_DEFINE_COMPONENT_ACTION(component_namespace, bind_prefix);
    HPX_DEFINE_COMPONENT_ACTION(component_namespace, bind_name);
    HPX_DEFINE_COMPONENT_ACTION(component_namespace, resolve_id);
    HPX_DEFINE_COMPONENT_ACTION(component_namespace, unbind);
    HPX_DEFINE_COMPONENT_ACTION(component_namespace, iterate_types);
    HPX_DEFINE_COMPONENT_ACTION(component_namespace, get_component_type_name);
    HPX_DEFINE_COMPONENT_ACTION(component_namespace, get_num_localities);
    HPX_DEFINE_COMPONENT_ACTION(component_namespace, statistics_counter);
};

}}}

HPX_REGISTER_ACTION_DECLARATION(
    hpx::agas::server::component_namespace::bind_prefix_action,
    component_namespace_bind_prefix_action)

HPX_REGISTER_ACTION_DECLARATION(
    hpx::agas::server::component_namespace::bind_name_action,
    component_namespace_bind_name_action)

HPX_REGISTER_ACTION_DECLARATION(
    hpx::agas::server::component_namespace::resolve_id_action,
    component_namespace_resolve_id_action)

HPX_REGISTER_ACTION_DECLARATION(
    hpx::agas::server::component_namespace::unbind_action,
    component_namespace_unbind_action)

HPX_REGISTER_ACTION_DECLARATION(
    hpx::agas::server::component_namespace::iterate_types_action,
    component_namespace_iterate_types_action)

HPX_REGISTER_ACTION_DECLARATION(
    hpx::agas::server::component_namespace::get_component_type_name_action,
    component_namespace_get_component_type_action)

HPX_REGISTER_ACTION_DECLARATION(
    hpx::agas::server::component_namespace::get_num_localities_action,
    component_namespace_get_num_localities_action)

HPX_REGISTER_ACTION_DECLARATION(
    hpx::agas::server::component_namespace::statistics_counter_action,
    component_namespace_statistics_counter_action)


#endif // HPX_A16135FC_AA32_444F_BB46_549AD456A661

