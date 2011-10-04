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
#include <hpx/runtime/components/component_type.hpp>
#include <hpx/runtime/components/server/fixed_component_base.hpp>
#include <hpx/util/insert_checked.hpp>
#include <hpx/util/logging.hpp>
#include <hpx/util/spinlock.hpp>

namespace hpx { namespace agas
{

HPX_EXPORT naming::gid_type bootstrap_component_namespace_gid(); 
HPX_EXPORT naming::id_type bootstrap_component_namespace_id(); 

namespace server
{

struct component_namespace :
    components::fixed_component_base<
        HPX_AGAS_COMPONENT_NS_MSB, HPX_AGAS_COMPONENT_NS_LSB, // constant GID
        component_namespace
    >
{
    // {{{ nested types
    typedef util::spinlock database_mutex_type;

    typedef int component_id_type;
    typedef boost::uint32_t prefix_type;

    typedef std::set<prefix_type> prefixes_type;

    typedef std::map<std::string, component_id_type> component_id_table_type; 

    typedef std::map<component_id_type, prefixes_type> factory_table_type;
    // }}}

  private:
    database_mutex_type mutex_;
    component_id_table_type component_ids_;
    factory_table_type factories_;
    component_id_type type_counter; 
 
  public:
    component_namespace()
      : mutex_()
      , component_ids_()
      , factories_()
      , type_counter(components::component_first_dynamic)
    {}

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

    response bind_prefix(
        std::string const& key
      , naming::gid_type const& prefix
        )
    {
        return bind_prefix(key, naming::get_prefix_from_gid(prefix), throws);
    }

    response bind_prefix(
        std::string const& key
      , prefix_type prefix
        )
    {
        return bind_prefix(key, prefix, throws);
    }

    response bind_prefix(
        std::string const& key
      , naming::gid_type const& prefix
      , error_code& ec
        )
    {
        return bind_prefix(key, naming::get_prefix_from_gid(prefix), ec);
    }

    response bind_prefix(
        std::string const& key
      , prefix_type prefix
      , error_code& ec
        );

    response bind_name(
        std::string const& key
        )
    {
        return bind_name(key, throws);
    }
    
    response bind_name(
        std::string const& key
      , error_code& ec
        );

    response resolve_id(
        components::component_type key
        )
    {
        return resolve_id(component_id_type(key), throws);
    }

    response resolve_id(
        component_id_type key
        )
    {
        return resolve_id(key, throws);
    }

    response resolve_id(
        components::component_type key
      , error_code& ec
        )
    {
        return resolve_id(component_id_type(key), ec);
    }

    response resolve_id(
        component_id_type key
      , error_code& ec
        );

    response resolve_name(
        std::string const& key
        )
    {
        return resolve_name(key, throws);
    }
    
    response resolve_name(
        std::string const& key
      , error_code& ec
        );

    response unbind(
        std::string const& key
        )
    {
        return unbind(key, throws);
    }
    
    response unbind(
        std::string const& key
      , error_code& ec
        );

    enum actions
    { // {{{ action enum
        namespace_bind_prefix  = BOOST_BINARY_U(0100000),
        namespace_bind_name    = BOOST_BINARY_U(0100001),
        namespace_resolve_id   = BOOST_BINARY_U(0100010),
        namespace_resolve_name = BOOST_BINARY_U(0100011),
        namespace_unbind       = BOOST_BINARY_U(0100100),
        namespace_service      = BOOST_BINARY_U(0100101)
    }; // }}}

    typedef hpx::actions::result_action1<
        component_namespace,
        /* return type */ response,
        /* enum value */  namespace_service,
        /* arguments */   request const&,
        &component_namespace::service
      , threads::thread_priority_critical
    > service_action;
    
    typedef hpx::actions::result_action2<
        component_namespace,
        /* return type */ response,
        /* enum value */  namespace_bind_prefix,
        /* arguments */   std::string const&, prefix_type,
        &component_namespace::bind_prefix
      , threads::thread_priority_critical
    > bind_prefix_action;
    
    typedef hpx::actions::result_action1<
        component_namespace,
        /* return type */ response,
        /* enum value */  namespace_bind_name,
        /* arguments */   std::string const&,
        &component_namespace::bind_name
      , threads::thread_priority_critical
    > bind_name_action;
    
    typedef hpx::actions::result_action1<
        component_namespace,
        /* return type */ response,
        /* enum value */  namespace_resolve_id,
        /* arguments */   component_id_type,
        &component_namespace::resolve_id
      , threads::thread_priority_critical
    > resolve_id_action;
    
    typedef hpx::actions::result_action1<
        component_namespace,
        /* return type */ response,
        /* enum value */  namespace_resolve_name,
        /* arguments */   std::string const&,
        &component_namespace::resolve_name
      , threads::thread_priority_critical
    > resolve_name_action;
    
    typedef hpx::actions::result_action1<
        component_namespace,
        /* return type */ response,
        /* enum value */  namespace_unbind,
        /* arguments */   std::string const&,
        &component_namespace::unbind
      , threads::thread_priority_critical
    > unbind_action;
};

}}}

#endif // HPX_A16135FC_AA32_444F_BB46_549AD456A661

