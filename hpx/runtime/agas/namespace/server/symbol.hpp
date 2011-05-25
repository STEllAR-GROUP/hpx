////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(HPX_D69CE952_C5D9_4545_B83E_BA3DCFD812EB)
#define HPX_D69CE952_C5D9_4545_B83E_BA3DCFD812EB

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/components/server/fixed_component_base.hpp>
#include <hpx/runtime/components/server/simple_component_base.hpp>
#include <hpx/runtime/agas/traits.hpp>
#include <hpx/runtime/agas/database/table.hpp>

namespace hpx { namespace agas { namespace server
{

template <typename Base, typename Database>
struct HPX_COMPONENT_EXPORT symbol_namespace_base : Base 
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
    symbol_namespace_base(std::string const& name)
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
        symbol_namespace_base<Base, Database>,
        /* return type */ bool,
        /* enum value */  namespace_bind,
        /* arguments */   symbol_type const&, naming::gid_type const&,
        &symbol_namespace_base<Base, Database>::bind
    > bind_action;
    
    typedef hpx::actions::result_action2<
        symbol_namespace_base<Base, Database>,
        /* return type */ naming::gid_type,
        /* enum value */  namespace_rebind,
        /* arguments */   symbol_type const&, naming::gid_type const&,
        &symbol_namespace_base<Base, Database>::rebind
    > rebind_action;
    
    typedef hpx::actions::result_action1<
        symbol_namespace_base<Base, Database>,
        /* return type */ naming::gid_type,
        /* enum value */  namespace_resolve,
        /* arguments */   symbol_type const&,
        &symbol_namespace_base<Base, Database>::resolve
    > resolve_action;
    
    typedef hpx::actions::result_action1<
        symbol_namespace_base<Base, Database>,
        /* retrun type */ bool,
        /* enum value */  namespace_unbind,
        /* arguments */   symbol_type const&,
        &symbol_namespace_base<Base, Database>::unbind
    > unbind_action;

    #if 0
    template <typename Derived>
    struct action_types
    { // {{{ action rebinder
        typedef hpx::actions::result_action2<
            Derived,
            /* return type */ bool,
            /* enum value */  namespace_bind,
            /* arguments */   symbol_type const&, naming::gid_type const&,
            &Derived::bind
        > bind;
        
        typedef hpx::actions::result_action2<
            Derived,
            /* return type */ naming::gid_type,
            /* enum value */  namespace_rebind,
            /* arguments */   symbol_type const&, naming::gid_type const&,
            &Derived::rebind
        > rebind;
        
        typedef hpx::actions::result_action1<
            Derived,
            /* return type */ naming::gid_type,
            /* enum value */  namespace_resolve,
            /* arguments */   symbol_type const&,
            &Derived::resolve
        > resolve;
        
        typedef hpx::actions::result_action1<
            Derived,
            /* retrun type */ bool,
            /* enum value */  namespace_unbind,
            /* arguments */   symbol_type const&,
            &Derived::unbind
        > unbind;
    }; // }}}
    #endif
};

template <typename Database>
struct HPX_COMPONENT_EXPORT bootstrap_symbol_namespace
  : symbol_namespace_base<
        components::fixed_component_base<
            HPX_AGAS_SYMBOL_NS_MSB, HPX_AGAS_SYMBOL_NS_LSB, // constant GID
            bootstrap_symbol_namespace<Database> >,
        Database>
{
    typedef symbol_namespace_base<
        components::fixed_component_base<
            HPX_AGAS_SYMBOL_NS_MSB, HPX_AGAS_SYMBOL_NS_LSB, // constant GID
            bootstrap_symbol_namespace<Database> >,
        Database
    > base_type;

    #if 0
    typedef typename base_type::template
      action_types<bootstrap_symbol_namespace> bound_action_types;

    typedef typename bound_action_types::bind bind_action;
    typedef typename bound_action_types::rebind rebind_action;
    typedef typename bound_action_types::resolve resolve_action;
    typedef typename bound_action_types::unbind unbind_action;
    #endif

    bootstrap_symbol_namespace():
      base_type("bootstrap_symbol_namespace") {} 
};

template <typename Database>
struct HPX_COMPONENT_EXPORT symbol_namespace
  : symbol_namespace_base<
        components::simple_component_base<symbol_namespace<Database> >,
        Database>
{
    typedef symbol_namespace_base<
        components::simple_component_base<symbol_namespace<Database> >,
        Database
    > base_type;

    #if 0
    typedef typename base_type::template
      action_types<symbol_namespace> bound_action_types;

    typedef typename bound_action_types::bind bind_action;
    typedef typename bound_action_types::rebind rebind_action;
    typedef typename bound_action_types::resolve resolve_action;
    typedef typename bound_action_types::unbind unbind_action;
    #endif

    symbol_namespace(): base_type("symbol_namespace") {} 
};

}}}

#endif // HPX_D69CE952_C5D9_4545_B83E_BA3DCFD812EB

