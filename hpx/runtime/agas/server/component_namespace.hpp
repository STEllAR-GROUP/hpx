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
#include <hpx/exception_fwd.hpp>
#include <hpx/lcos/local/mutex.hpp>
#include <hpx/lcos/local/spinlock.hpp>
#include <hpx/runtime/agas/namespace_action_code.hpp>
#include <hpx/runtime/agas/request.hpp>
#include <hpx/runtime/agas/response.hpp>
#include <hpx/runtime/components/component_type.hpp>
#include <hpx/runtime/components/server/fixed_component_base.hpp>
#include <hpx/runtime/serialization/vector.hpp>
#include <hpx/util/function.hpp>
#include <hpx/util/high_resolution_clock.hpp>
#include <hpx/util/insert_checked.hpp>
#include <hpx/util/logging.hpp>

#include <map>
#include <set>
#include <string>
#include <vector>

#include <boost/atomic.hpp>
#include <boost/bimap.hpp>
#include <boost/format.hpp>

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

    typedef hpx::util::function<
        void(std::string const&, components::component_type)
    > iterate_types_function_type;

    typedef components::component_type component_id_type;

    typedef std::set<boost::uint32_t> prefixes_type;

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

    struct update_time_on_exit;

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

            boost::atomic<boost::int64_t> count_;
            boost::atomic<boost::int64_t> time_;
        };

        counter_data()
        {}

    public:
        // access current counter values
        boost::int64_t get_bind_prefix_count(bool);
        boost::int64_t get_bind_name_count(bool);
        boost::int64_t get_resolve_id_count(bool);
        boost::int64_t get_unbind_name_count(bool);
        boost::int64_t get_iterate_types_count(bool);
        boost::int64_t get_component_type_name_count(bool);
        boost::int64_t get_num_localities_count(bool);
        boost::int64_t get_overall_count(bool);

        boost::int64_t get_bind_prefix_time(bool);
        boost::int64_t get_bind_name_time(bool);
        boost::int64_t get_resolve_id_time(bool);
        boost::int64_t get_unbind_name_time(bool);
        boost::int64_t get_iterate_types_time(bool);
        boost::int64_t get_component_type_name_time(bool);
        boost::int64_t get_num_localities_time(bool);
        boost::int64_t get_overall_time(bool);

        // increment counter values
        void increment_bind_prefix_count();
        void increment_bind_name_count();
        void increment_resolve_id_count();
        void increment_unbind_name_ount();
        void increment_iterate_types_count();
        void increment_get_component_type_name_count();
        void increment_num_localities_count();

    private:
        friend struct update_time_on_exit;
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

    struct update_time_on_exit
    {
        update_time_on_exit(boost::atomic<boost::int64_t>& t)
          : started_at_(hpx::util::high_resolution_clock::now())
          , t_(t)
        {}

        ~update_time_on_exit()
        {
            t_ += (hpx::util::high_resolution_clock::now() - started_at_);
        }

        boost::uint64_t started_at_;
        boost::atomic<boost::int64_t>& t_;
    };

  public:
    component_namespace()
      : base_type(HPX_AGAS_COMPONENT_NS_MSB, HPX_AGAS_COMPONENT_NS_LSB)
      , type_counter(components::component_first_dynamic)
    {}

    void finalize();

    response remote_service(
        request const& req
        )
    {
        return service(req, throws);
    }

    response service(
        request const& req
      , error_code& ec
        );

    /// Maps \a service over \p reqs in parallel.
    std::vector<response> remote_bulk_service(
        std::vector<request> const& reqs
        )
    {
        return bulk_service(reqs, throws);
    }

    /// Maps \a service over \p reqs in parallel.
    std::vector<response> bulk_service(
        std::vector<request> const& reqs
      , error_code& ec
        );

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

    response bind_prefix(
        request const& req
      , error_code& ec = throws
        );

    response bind_name(
        request const& req
      , error_code& ec = throws
        );

    response resolve_id(
        request const& req
      , error_code& ec = throws
        );

    response unbind(
        request const& req
      , error_code& ec = throws
        );

    response iterate_types(
        request const& req
      , error_code& ec = throws
        );

    response get_component_type_name(
        request const& req
      , error_code& ec = throws
        );

    response get_num_localities(
        request const& req
      , error_code& ec = throws
        );

    response statistics_counter(
        request const& req
      , error_code& ec = throws
        );

    enum actions
    { // {{{ action enum
        // Actual actions
        namespace_service                 = component_ns_service
      , namespace_bulk_service            = component_ns_bulk_service

        // Pseudo-actions
      , namespace_bind_prefix             = component_ns_bind_prefix
      , namespace_bind_name               = component_ns_bind_name
      , namespace_resolve_id              = component_ns_resolve_id
      , namespace_unbind_name             = component_ns_unbind_name
      , namespace_iterate_types           = component_ns_iterate_types
      , namespace_get_component_type_name = component_ns_get_component_type_name
      , namespace_num_localities     = component_ns_num_localities
      , namespace_statistics              = component_ns_statistics_counter
    }; // }}}

    HPX_DEFINE_COMPONENT_ACTION(component_namespace, remote_service, service_action);
    HPX_DEFINE_COMPONENT_ACTION(component_namespace, remote_bulk_service,
        bulk_service_action);
};

}}}

HPX_ACTION_USES_MEDIUM_STACK(
    hpx::agas::server::component_namespace::service_action)

HPX_ACTION_USES_MEDIUM_STACK(
    hpx::agas::server::component_namespace::bulk_service_action)

HPX_REGISTER_ACTION_DECLARATION(
    hpx::agas::server::component_namespace::service_action,
    component_namespace_service_action)

HPX_REGISTER_ACTION_DECLARATION(
    hpx::agas::server::component_namespace::bulk_service_action,
    component_namespace_bulk_service_action)

#endif // HPX_A16135FC_AA32_444F_BB46_549AD456A661

