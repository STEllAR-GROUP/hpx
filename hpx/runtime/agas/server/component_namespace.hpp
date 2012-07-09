////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Adelstein-Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(HPX_A16135FC_AA32_444F_BB46_549AD456A661)
#define HPX_A16135FC_AA32_444F_BB46_549AD456A661

#include <map>
#include <set>

#include <boost/format.hpp>
#include <boost/assert.hpp>
#include <boost/utility/binary.hpp>

#include <hpx/hpx_fwd.hpp>
#include <hpx/exception.hpp>
#include <hpx/runtime/agas/request.hpp>
#include <hpx/runtime/agas/response.hpp>
#include <hpx/runtime/agas/namespace_action_code.hpp>
#include <hpx/runtime/components/component_type.hpp>
#include <hpx/runtime/components/server/fixed_component_base.hpp>
#include <hpx/util/insert_checked.hpp>
#include <hpx/util/logging.hpp>
#include <hpx/util/function.hpp>
#include <hpx/lcos/local/mutex.hpp>
#include <hpx/lcos/local/spinlock.hpp>

namespace hpx { namespace agas
{

HPX_EXPORT naming::gid_type bootstrap_component_namespace_gid();
HPX_EXPORT naming::id_type bootstrap_component_namespace_id();

namespace server
{

// Base name used to register the component
char const* const component_namespace_service_name = "/component_namespace/";

struct HPX_EXPORT component_namespace :
    components::fixed_component_base<
        HPX_AGAS_COMPONENT_NS_MSB, HPX_AGAS_COMPONENT_NS_LSB, // constant GID
        component_namespace
    >
{
    // {{{ nested types
    typedef lcos::local::mutex mutex_type;

    typedef hpx::util::function<
        void(std::string const&, components::component_type)
    > iterate_types_function_type;

    typedef components::component_type component_id_type;

    typedef std::set<boost::uint32_t> prefixes_type;

    typedef std::map<std::string, component_id_type> component_id_table_type;

    typedef std::map<component_id_type, prefixes_type> factory_table_type;
    // }}}

  private:
    // REVIEW: Separate mutexes might reduce contention here. This has to be
    // investigated carefully.
    mutex_type mutex_;
    component_id_table_type component_ids_;
    factory_table_type factories_;
    component_id_type type_counter;
    std::string instance_name;

    // data structure holding all counters for the omponent_namespace component
    struct counter_data :  boost::noncopyable
    {
      typedef lcos::local::spinlock mutex_type;

      counter_data()
        : bind_prefix_(0)
        , bind_name_(0)
        , resolve_id_(0)
        , unbind_(0)
        , iterate_types_(0)
      {}

    public:
      // access current counter values
      boost::int64_t get_bind_prefix_count() const;
      boost::int64_t get_bind_name_count() const;
      boost::int64_t get_resolve_id_count() const;
      boost::int64_t get_unbind_count() const;
      boost::int64_t get_iterate_types_count() const;

      // increment counter values
      void increment_bind_prefix_count();
      void increment_bind_name_count();
      void increment_resolve_id_count();
      void increment_unbind_count();
      void increment_iterate_types_count();

      mutable mutex_type mtx_;
      boost::int64_t bind_prefix_;          // component_ns_bind_prefix
      boost::int64_t bind_name_;            // component_ns_bind_name
      boost::int64_t resolve_id_;           // component_ns_resolve_id
      boost::int64_t unbind_;               // component_ns_unbind
      boost::int64_t iterate_types_;        // component_ns_iterate_types
    };
    counter_data counter_data_;

  public:
    component_namespace()
      : mutex_()
      , component_ids_()
      , factories_()
      , type_counter(components::component_first_dynamic)
    {}

    void finalize();

    response service(
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
    std::vector<response> bulk_service(
        std::vector<request> const& reqs
        )
    {
        return bulk_service(reqs, throws);
    }

    // register all performance counter types exposed by this component
    void register_counter_types(
        char const* servicename
      , error_code& ec = throws);

    /// Maps \a service over \p reqs in parallel.
    std::vector<response> bulk_service(
        std::vector<request> const& reqs
      , error_code& ec
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

    response statistics_counter(
        request const& req
      , error_code& ec = throws
        );

    enum actions
    { // {{{ action enum
        // Actual actions
        namespace_service       = component_ns_service
      , namespace_bulk_service  = component_ns_bulk_service

        // Pseudo-actions
      , namespace_bind_prefix   = component_ns_bind_prefix
      , namespace_bind_name     = component_ns_bind_name
      , namespace_resolve_id    = component_ns_resolve_id
      , namespace_unbind        = component_ns_unbind
      , namespace_iterate_types = component_ns_iterate_types
      , namespace_statistics    = component_ns_statistics_counter
    }; // }}}

    typedef hpx::actions::result_action1<
        component_namespace
      , /* return type */ response
      , /* enum value */  namespace_service
      , /* arguments */   request const&
      , &component_namespace::service
      , threads::thread_priority_critical
    > service_action;

    typedef hpx::actions::result_action1<
        component_namespace
      , /* return type */ std::vector<response>
      , /* enum value */  namespace_bulk_service
      , /* arguments */   std::vector<request> const&
      , &component_namespace::bulk_service
      , threads::thread_priority_critical
    > bulk_service_action;
};

}}}

HPX_REGISTER_ACTION_DECLARATION_EX(
    hpx::agas::server::component_namespace::service_action,
    component_namespace_service_action)

HPX_REGISTER_ACTION_DECLARATION_EX(
    hpx::agas::server::component_namespace::bulk_service_action,
    component_namespace_bulk_service_action)

#endif // HPX_A16135FC_AA32_444F_BB46_549AD456A661

