////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Adelstein-Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(HPX_D69CE952_C5D9_4545_B83E_BA3DCFD812EB)
#define HPX_D69CE952_C5D9_4545_B83E_BA3DCFD812EB

#include <map>

#include <boost/format.hpp>
#include <boost/utility/binary.hpp>

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/actions/function.hpp>
#include <hpx/runtime/agas/request.hpp>
#include <hpx/runtime/agas/response.hpp>
#include <hpx/runtime/components/server/fixed_component_base.hpp>
#include <hpx/util/insert_checked.hpp>
#include <hpx/util/logging.hpp>
#include <hpx/util/spinlock.hpp>

namespace hpx { namespace agas
{

HPX_EXPORT naming::gid_type bootstrap_symbol_namespace_gid(); 
HPX_EXPORT naming::id_type bootstrap_symbol_namespace_id(); 

namespace server
{

struct symbol_namespace :
    components::fixed_component_base<
        HPX_AGAS_SYMBOL_NS_MSB, HPX_AGAS_SYMBOL_NS_LSB, // constant GID
        symbol_namespace
    >
{
    // {{{ nested types
    typedef util::spinlock database_mutex_type;

    typedef hpx::actions::function<
        void(std::string const&, naming::gid_type const&)
    > iterate_symbols_function_type;

    typedef std::map<std::string, naming::gid_type> gid_table_type; 
    // }}} 
 
  private:
    database_mutex_type mutex_;
    gid_table_type gids_;
  
  public:
    symbol_namespace()
      : mutex_()
      , gids_()
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
    
    response bind(
        std::string const& key
      , naming::gid_type const& gid
        )
    {
        return bind(key, gid, throws);
    }

    response bind(
        std::string const& key
      , naming::gid_type const& gid
      , error_code& ec
        );

    response resolve(
        std::string const& key
        )
    {
        return resolve(key, throws);
    }

    response resolve(
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

    response iterate(
        iterate_symbols_function_type const& f
        )
    {
        return iterate(f, throws);
    }

    response iterate(
        iterate_symbols_function_type const& f
      , error_code& ec
        );

    enum actions
    { // {{{ action enum
        namespace_bind    = BOOST_BINARY_U(0010000),
        namespace_resolve = BOOST_BINARY_U(0010001),
        namespace_unbind  = BOOST_BINARY_U(0010010),
        namespace_iterate = BOOST_BINARY_U(0010011),
        namespace_service = BOOST_BINARY_U(0010100)
    }; // }}}

    typedef hpx::actions::result_action1<
        symbol_namespace,
        /* return type */ response,
        /* enum value */  namespace_service,
        /* arguments */   request const&,
        &symbol_namespace::service
      , threads::thread_priority_critical
    > service_action;
    
    typedef hpx::actions::result_action2<
        symbol_namespace,
        /* return type */ response,
        /* enum value */  namespace_bind,
        /* arguments */   std::string const&, naming::gid_type const&,
        &symbol_namespace::bind
      , threads::thread_priority_critical
    > bind_action;
    
    typedef hpx::actions::result_action1<
        symbol_namespace,
        /* return type */ response,
        /* enum value */  namespace_resolve,
        /* arguments */   std::string const&,
        &symbol_namespace::resolve
      , threads::thread_priority_critical
    > resolve_action;
    
    typedef hpx::actions::result_action1<
        symbol_namespace,
        /* retrun type */ response,
        /* enum value */  namespace_unbind,
        /* arguments */   std::string const&,
        &symbol_namespace::unbind
      , threads::thread_priority_critical
    > unbind_action;

    typedef hpx::actions::result_action1<
        symbol_namespace,
        /* retrun type */ response,
        /* enum value */  namespace_iterate,
        /* arguments */   iterate_symbols_function_type const&,
        &symbol_namespace::iterate
      , threads::thread_priority_critical
    > iterate_action;
};

}}}

#endif // HPX_D69CE952_C5D9_4545_B83E_BA3DCFD812EB

