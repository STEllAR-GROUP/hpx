////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(HPX_A16135FC_AA32_444F_BB46_549AD456A661)
#define HPX_A16135FC_AA32_444F_BB46_549AD456A661

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/components/server/simple_component_base.hpp>
#include <hpx/runtime/agas/traits.hpp>
#include <hpx/runtime/agas/database/table.hpp>

namespace hpx { namespace agas { namespace server
{

template <typename Database>
struct HPX_COMPONENT_EXPORT component_namespace
  : simple_component_base<component_namespace<Database> >
{
    typedef typename traits::database::mutex_type<Database>::type
        database_mutex_type;

    typedef std::string component_name_type;
    typedef int component_id_type;
    typedef boost::uint32_t prefix_type
    typedef boost::unordered_set<prefix_type> prefixes_type;

    typedef table<Database, component_name_type, component_id_type>
        component_id_table_type; 

    typedef table<Database, component_id_type, prefixes_type>
        factory_table_type;

    enum actions
    {
        namespace_bind,
        namespace_resolve_name,
        namespace_resolve_id,
        namespace_unbind,
    };
  
  private:
    database_mutex_type mutex_;
    component_id_table_type component_ids_;
    factory_table_type factories_;
  
  public:
    component_namespace()
      : mutex_(),
        component_ids_("hpx.agas.component_namespace.id"),
        factories_("hpx.agas.component_namespace.factory")
    { traits::initialize_mutex(mutex_); }

    component_id_type bind(component_name_type const& key, prefix_type prefix)
    {
        typename database_mutex_type::scoped_lock l(mutex_);
        // TODO: implement
    }

    prefixes_type resolve_id(component_id_type key)
    {
        typename database_mutex_type::scoped_lock l(mutex_);
        // TODO: implement
    } 
    
    component_id_type resolve_name(component_name_type const& key)
    {
        typename database_mutex_type::scoped_lock l(mutex_);
        // TODO: implement
    } 
    
    void unbind(component_name_type const& key)
    {
        typename database_mutex_type::scoped_lock l(mutex_);
        // TODO: implement
    } 

    typedef hpx::actions::result_action2<
        component_namespace<Database>,
        component_id_type,                               // return type
        namespace_bind,                                  // action type
        component_name_type const&, prefix_type,         // arguments 
        &component_namespace<Database>::bind
    > bind_action;
    
    typedef hpx::actions::result_action1<
        component_namespace<Database>,
        prefixes_type,                                   // return type
        namespace_resolve_id,                            // action type
        component_id_type,                               // arguments 
        &component_namespace<Database>::resolve_id
    > resolve_id_action;
    
    typedef hpx::actions::result_action1<
        component_namespace<Database>,
        component_id_type,                               // return type
        namespace_resolve_name,                          // action type
        component_name_type const&,                      // arguments 
        &component_namespace<Database>::resolve_name
    > resolve_name_action;
    
    typedef hpx::actions::result_action1<
        component_namespace<Database>,
        void,                                            // return type
        namespace_unbind,                                // action type
        component_name_type const&,                      // arguments 
        &component_namespace<Database>::unbind
    > unbind_action;
};

}}}}

#endif // HPX_A16135FC_AA32_444F_BB46_549AD456A661

