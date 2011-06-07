////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(HPX_D69CE952_C5D9_4545_B83E_BA3DCFD812EB)
#define HPX_D69CE952_C5D9_4545_B83E_BA3DCFD812EB

#include <boost/utility/binary.hpp>

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/agas/traits.hpp>
#include <hpx/runtime/agas/database/table.hpp>
#include <hpx/runtime/agas/namespace/response.hpp>
#include <hpx/runtime/components/server/fixed_component_base.hpp>

// TODO: use response move semantics (?)

namespace hpx { namespace agas { namespace server
{

template <typename Database, typename Protocol>
struct HPX_COMPONENT_EXPORT symbol_namespace :
  components::fixed_component_base<
    HPX_AGAS_SYMBOL_NS_MSB, HPX_AGAS_SYMBOL_NS_LSB, // constant GID
    symbol_namespace<Database, Protocol>
  >
{
    // {{{ nested types
    typedef typename traits::database::mutex_type<Database>::type
        database_mutex_type;

    typedef std::string symbol_type;

    typedef response<Protocol> response_type;

    typedef table<Database, symbol_type, naming::gid_type>
        gid_table_type; 
    // }}} 
 
  private:
    database_mutex_type mutex_;
    gid_table_type gids_;
  
  public:
    symbol_namespace(std::string const& name = "root_symbol_namespace")
      : mutex_(),
        gids_(std::string("hpx.agas.") + name + ".gid")
    { traits::initialize_mutex(mutex_); }

    response_type bind(
        symbol_type const& key
      , naming::gid_type const& gid
    ) { // {{{ bind implementation
        typename database_mutex_type::scoped_lock l(mutex_);

        // Always load the table once, as this operation might be slow for some
        // database backends.
        typename gid_table_type::map_type& gid_table = gids_.get();

        typename gid_table_type::map_type::iterator
            it = gid_table.find(key), end = gid_table.end();

        if (it != end)
            return response_type(symbol_ns_bind, no_success);

        gid_table.insert(typename
            gid_table_type::map_type::value_type(key, gid));
        return response_type(symbol_ns_bind); 
    } // }}}
    
    response_type rebind(
        symbol_type const& key, naming::gid_type const& gid
    ) { // {{{ bind implementation
        typename database_mutex_type::scoped_lock l(mutex_);

        // Always load the table once, as this operation might be slow for some
        // database backends.
        typename gid_table_type::map_type& gid_table = gids_.get();

        typename gid_table_type::map_type::iterator
            it = gid_table.find(key), end = gid_table.end();

        if (it != end)
        {
            naming::gid_type old = it->second;
            it->second = gid;
            return response_type(symbol_ns_rebind, old);
        }

        gid_table.insert(typename
            gid_table_type::map_type::value_type(key, gid));
        return response_type(symbol_ns_rebind, gid); 
    } // }}}

    response_type resolve(
        symbol_type const& key
    ) { // {{{ resolve implementation
        typename database_mutex_type::scoped_lock l(mutex_);

        typename gid_table_type::map_type& gid_table = gids_.get();

        typename gid_table_type::map_type::iterator
            it = gid_table.find(key), end = gid_table.end();

        if (it == end)
            return response_type(symbol_ns_resolve
                               , naming::invalid_gid
                               , no_success);

        return response_type(symbol_ns_resolve, it->second);
    } // }}}  
    
    response_type unbind(
        symbol_type const& key
    ) { // {{{ unbind implementation
        typename database_mutex_type::scoped_lock l(mutex_);
        
        typename gid_table_type::map_type& gid_table = gids_.get();

        typename gid_table_type::map_type::iterator
            it = gid_table.find(key), end = gid_table.end();

        if (it == end)
            return response_type(symbol_ns_unbind, no_success);

        gid_table.erase(key);
        return response_type(symbol_ns_unbind);
    } // }}} 

    enum actions
    { // {{{ action enum
        namespace_bind    = BOOST_BINARY_U(0010000),
        namespace_rebind  = BOOST_BINARY_U(0010001),
        namespace_resolve = BOOST_BINARY_U(0010010),
        namespace_unbind  = BOOST_BINARY_U(0010011)
    }; // }}}
    
    typedef hpx::actions::result_action2<
        symbol_namespace<Database, Protocol>,
        /* return type */ response_type,
        /* enum value */  namespace_bind,
        /* arguments */   symbol_type const&, naming::gid_type const&,
        &symbol_namespace<Database, Protocol>::bind
    > bind_action;
    
    typedef hpx::actions::result_action2<
        symbol_namespace<Database, Protocol>,
        /* return type */ response_type,
        /* enum value */  namespace_rebind,
        /* arguments */   symbol_type const&, naming::gid_type const&,
        &symbol_namespace<Database, Protocol>::rebind
    > rebind_action;
    
    typedef hpx::actions::result_action1<
        symbol_namespace<Database, Protocol>,
        /* return type */ response_type,
        /* enum value */  namespace_resolve,
        /* arguments */   symbol_type const&,
        &symbol_namespace<Database, Protocol>::resolve
    > resolve_action;
    
    typedef hpx::actions::result_action1<
        symbol_namespace<Database, Protocol>,
        /* retrun type */ response_type,
        /* enum value */  namespace_unbind,
        /* arguments */   symbol_type const&,
        &symbol_namespace<Database, Protocol>::unbind
    > unbind_action;
};

}}}

#endif // HPX_D69CE952_C5D9_4545_B83E_BA3DCFD812EB

