////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(HPX_D69CE952_C5D9_4545_B83E_BA3DCFD812EB)
#define HPX_D69CE952_C5D9_4545_B83E_BA3DCFD812EB

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/agas/traits.hpp>
#include <hpx/runtime/agas/database/table.hpp>
#include <hpx/runtime/components/server/fixed_component_base.hpp>

namespace hpx { namespace agas { namespace server
{

template <typename Database>
struct HPX_COMPONENT_EXPORT symbol_namespace :
  components::fixed_component_base<
    HPX_AGAS_SYMBOL_NS_MSB, HPX_AGAS_SYMBOL_NS_LSB, // constant GID
    primary_namespace<Database, Protocol>
  >
{
    // {{{ nested types
    typedef typename traits::database::mutex_type<Database>::type
        database_mutex_type;

    typedef std::string symbol_type;

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

    bool bind(symbol_type const& key, naming::gid_type const& gid)
    { // {{{ bind implementation
        typename database_mutex_type::scoped_lock l(mutex_);

        // Always load the table once, as this operation might be slow for some
        // database backends.
        typename gid_table_type::map_type& gid_table = gids_.get();

        typename gid_table_type::map_type::iterator
            it = gid_table.find(key), end = gid_table.end();

        if (it != end)
            return false;

        gid_table.insert(typename
            gid_table_type::map_type::value_type(key, gid));
        return true; 
    } // }}}
    
    naming::gid_type rebind(symbol_type const& key, naming::gid_type const& gid)
    { // {{{ bind implementation
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
            return old;
        }

        gid_table.insert(typename
            gid_table_type::map_type::value_type(key, gid));
        return gid; 
    } // }}}

    naming::gid_type resolve(symbol_type const& key)
    { // {{{ resolve implementation
        typename database_mutex_type::scoped_lock l(mutex_);

        typename gid_table_type::map_type& gid_table = gids_.get();

        typename gid_table_type::map_type::iterator
            it = gid_table.find(key), end = gid_table.end();

        if (it == end)
            return naming::invalid_gid;

        return it->second;
    } // }}}  
    
    bool unbind(symbol_type const& key)
    { // {{{ unbind implementation
        typename database_mutex_type::scoped_lock l(mutex_);
        
        typename gid_table_type::map_type& gid_table = gids_.get();

        typename gid_table_type::map_type::iterator
            it = gid_table.find(key), end = gid_table.end();

        if (it == end)
            return false;

        gid_table.erase(key);
        return true;
    } // }}} 

    enum actions
    { // {{{ action enum
        namespace_bind,
        namespace_rebind,
        namespace_resolve,
        namespace_unbind,
    }; // }}}
    
    typedef hpx::actions::result_action2<
        symbol_namespace<Database>,
        /* return type */ bool,
        /* enum value */  namespace_bind,
        /* arguments */   symbol_type const&, naming::gid_type const&,
        &symbol_namespace<Database>::bind
    > bind_action;
    
    typedef hpx::actions::result_action2<
        symbol_namespace<Database>,
        /* return type */ naming::gid_type,
        /* enum value */  namespace_rebind,
        /* arguments */   symbol_type const&, naming::gid_type const&,
        &symbol_namespace<Database>::rebind
    > rebind_action;
    
    typedef hpx::actions::result_action1<
        symbol_namespace<Database>,
        /* return type */ naming::gid_type,
        /* enum value */  namespace_resolve,
        /* arguments */   symbol_type const&,
        &symbol_namespace<Database>::resolve
    > resolve_action;
    
    typedef hpx::actions::result_action1<
        symbol_namespace<Database>,
        /* retrun type */ bool,
        /* enum value */  namespace_unbind,
        /* arguments */   symbol_type const&,
        &symbol_namespace<Database>::unbind
    > unbind_action;
};

}}}

#endif // HPX_D69CE952_C5D9_4545_B83E_BA3DCFD812EB

