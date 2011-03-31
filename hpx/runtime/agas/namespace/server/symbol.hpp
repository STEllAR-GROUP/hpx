////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(HPX_D69CE952_C5D9_4545_B83E_BA3DCFD812EB)
#define HPX_D69CE952_C5D9_4545_B83E_BA3DCFD812EB

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/components/server/simple_component_base.hpp>
#include <hpx/runtime/agas/traits.hpp>

namespace hpx { namespace agas { namespace server
{

template <typename Database>
struct HPX_COMPONENT_EXPORT symbol_namespace
  : simple_component_base<symbol_namespace<Database> >
{
    typedef typename traits::database::mutex_type<Database>::type
        database_mutex_type;

    typedef std::string symbol_type;

    typedef table<Database, symbol_type, naming::gid_type>
        gid_table_type; 

    enum actions
    {
        namespace_bind,
        namespace_resolve,
        namespace_unbind,
    };
  
  private:
    database_mutex_type mutex_;
    gid_table_type gids_;
  
  public:
    symbol_namespace()
      : mutex_(),
        gids_("hpx.agas.symbol_namespace.gid")
    { traits::initialize_mutex(mutex_); }

    void bind(symbol_type const& key, naming::gid_type const& gid)
    {
        typename database_mutex_type::scoped_lock l(mutex_);
        // TODO: implement
    }

    naming::gid_type resolve(symbol_type const& key)
    {
        typename database_mutex_type::scoped_lock l(mutex_);
        // TODO: implement
    } 
    
    void unbind(symbol_type const& key)
    {
        typename database_mutex_type::scoped_lock l(mutex_);
        // TODO: implement
    } 

    typedef hpx::actions::result_action2<
        symbol_namespace<Database>,
        void,                                            // return type
        namespace_bind,                                  // action type
        symbol_type const&, naming::gid_type const&,     // arguments 
        &symbol_namespace<Database>::bind
    > bind_action;
    
    typedef hpx::actions::result_action1<
        symbol_namespace<Database>,
        naming::gid_type,                                // return type
        namespace_resolve_id,                            // action type
        symbol_type const&,                              // arguments 
        &symbol_namespace<Database>::resolve_id
    > resolve_action;
    
    typedef hpx::actions::result_action1<
        symbol_namespace<Database>,
        void,                                            // return type
        namespace_unbind,                                // action type
        symbol_type const&,                              // arguments 
        &symbol_namespace<Database>::unbind
    > unbind_action;
};

}}}}

#endif // HPX_D69CE952_C5D9_4545_B83E_BA3DCFD812EB

