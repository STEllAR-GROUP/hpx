////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Adelstein-Lelbach
//  Copyright (c) 2012-2013 Hartmut Kaiser
//  Copyright (c) 2016 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(HPX_D69CE952_C5D9_4545_B83E_BA3DCFD812EB)
#define HPX_D69CE952_C5D9_4545_B83E_BA3DCFD812EB

#include <hpx/config.hpp>
#include <hpx/lcos/local/spinlock.hpp>
#include <hpx/runtime/actions/component_action.hpp>
#include <hpx/runtime/components/server/fixed_component_base.hpp>
#include <hpx/util/function.hpp>

#include <boost/atomic.hpp>
#include <boost/format.hpp>

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace hpx { namespace agas
{

HPX_EXPORT naming::gid_type bootstrap_symbol_namespace_gid();
HPX_EXPORT naming::id_type bootstrap_symbol_namespace_id();

namespace server
{

// Base name used to register the component
char const* const symbol_namespace_service_name = "symbol/";

struct HPX_EXPORT symbol_namespace
  : components::fixed_component_base<symbol_namespace>
{
    // {{{ nested types
    typedef lcos::local::spinlock mutex_type;
    typedef components::fixed_component_base<symbol_namespace> base_type;

    // FIXME: This signature should use id_type, not gid_type
    typedef hpx::util::function<
        void(std::string const&, naming::gid_type const&)
    > iterate_names_function_type;

    typedef std::map<std::string, std::shared_ptr<naming::gid_type> >
        gid_table_type;

    typedef std::multimap<std::string, hpx::id_type> on_event_data_map_type;
    // }}}

  private:
    mutex_type mutex_;
    gid_table_type gids_;
    std::string instance_name_;
    on_event_data_map_type on_event_data_;

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
        std::int64_t get_bind_count(bool);
        std::int64_t get_resolve_count(bool);
        std::int64_t get_unbind_count(bool);
        std::int64_t get_iterate_names_count(bool);
        std::int64_t get_on_event_count(bool);
        std::int64_t get_overall_count(bool);

        std::int64_t get_bind_time(bool);
        std::int64_t get_resolve_time(bool);
        std::int64_t get_unbind_time(bool);
        std::int64_t get_iterate_names_time(bool);
        std::int64_t get_on_event_time(bool);
        std::int64_t get_overall_time(bool);

        // increment counter values
        void increment_bind_count();
        void increment_resolve_count();
        void increment_unbind_count();
        void increment_iterate_names_count();
        void increment_on_event_count();

    private:
        friend struct update_time_on_exit;
        friend struct symbol_namespace;

        api_counter_data bind_;               // symbol_ns_bind
        api_counter_data resolve_;            // symbol_ns_resolve
        api_counter_data unbind_;             // symbol_ns_unbind
        api_counter_data iterate_names_;      // symbol_ns_iterate_names
        api_counter_data on_event_;           // symbol_ns_on_event
    };
    counter_data counter_data_;

  public:
    symbol_namespace()
      : base_type(HPX_AGAS_SYMBOL_NS_MSB, HPX_AGAS_SYMBOL_NS_LSB)
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
      , std::uint32_t locality_id = naming::invalid_locality_id
      , error_code& ec = throws
        );

    void unregister_server_instance(
        error_code& ec = throws
        );

    bool bind(
        std::string key
      , naming::gid_type gid
        );

    naming::gid_type resolve(std::string const& key);

    naming::gid_type unbind(std::string const& key);

    void iterate(iterate_names_function_type const& f);

    bool on_event(
        std::string const& name
      , bool call_for_past_events
      , hpx::id_type lco
        );

    naming::gid_type statistics_counter(std::string const& key);

    HPX_DEFINE_COMPONENT_ACTION(symbol_namespace, bind);
    HPX_DEFINE_COMPONENT_ACTION(symbol_namespace, resolve);
    HPX_DEFINE_COMPONENT_ACTION(symbol_namespace, unbind);
    HPX_DEFINE_COMPONENT_ACTION(symbol_namespace, iterate);
    HPX_DEFINE_COMPONENT_ACTION(symbol_namespace, on_event);
    HPX_DEFINE_COMPONENT_ACTION(symbol_namespace, statistics_counter);
};

}}}

HPX_ACTION_USES_MEDIUM_STACK(
    hpx::agas::server::symbol_namespace::bind_action)

HPX_REGISTER_ACTION_DECLARATION(
    hpx::agas::server::symbol_namespace::bind_action,
    symbol_namespace_bind_action)

HPX_ACTION_USES_MEDIUM_STACK(
    hpx::agas::server::symbol_namespace::resolve_action)

HPX_REGISTER_ACTION_DECLARATION(
    hpx::agas::server::symbol_namespace::resolve_action,
    symbol_namespace_resolve_action)

HPX_ACTION_USES_MEDIUM_STACK(
    hpx::agas::server::symbol_namespace::unbind_action)

HPX_REGISTER_ACTION_DECLARATION(
    hpx::agas::server::symbol_namespace::unbind_action,
    symbol_namespace_unbind_action)

HPX_ACTION_USES_MEDIUM_STACK(
    hpx::agas::server::symbol_namespace::iterate_action)

HPX_REGISTER_ACTION_DECLARATION(
    hpx::agas::server::symbol_namespace::iterate_action,
    symbol_namespace_iterate_action)

HPX_ACTION_USES_MEDIUM_STACK(
    hpx::agas::server::symbol_namespace::on_event_action)

HPX_REGISTER_ACTION_DECLARATION(
    hpx::agas::server::symbol_namespace::on_event_action,
    symbol_namespace_on_event_action)

HPX_ACTION_USES_MEDIUM_STACK(
    hpx::agas::server::symbol_namespace::statistics_counter_action)

HPX_REGISTER_ACTION_DECLARATION(
    hpx::agas::server::symbol_namespace::statistics_counter_action,
    symbol_namespace_statistics_counter_action)

#endif // HPX_D69CE952_C5D9_4545_B83E_BA3DCFD812EB

