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
    // {{{ nested types
    typedef typename traits::database::mutex_type<Database>::type
        database_mutex_type;

    typedef std::string component_name_type;
    typedef int component_id_type;
    typedef boost::uint32_t prefix_type
    typedef boost::unordered_set<prefix_type> prefixes_type;

    typedef table<Database, component_name_type, component_id_type>
        component_id_table_type; 

    // FIXME: this shouldn't hold prefixes_type, it needs to hold a vector of
    // component_id_types (yes?).
    typedef table<Database, component_id_type, prefixes_type>
        factory_table_type;
    // }}}

  private:
    database_mutex_type mutex_;
    component_id_table_type component_ids_;
    factory_table_type factories_;
    component_id_type type_counter; 
 
  public:
    component_namespace()
      : mutex_(),
        component_ids_("hpx.agas.component_namespace.id"),
        factories_("hpx.agas.component_namespace.factory"),
        type_counter(components::component_first_dynamic)
    { traits::initialize_mutex(mutex_); }

    component_id_type bind(component_name_type const& key, prefix_type prefix)
    { // {{{ bind implementation
        typename database_mutex_type::scoped_lock l(mutex_);

        // always load the table once, as the operation might be expensive for
        // some backends and the compiler may not be able to optimize this away
        // if the database backend has implicit/builtin atomic semantics.
        typename component_id_table_type::map_type& c_id_table =
            component_ids_.get();
        
        typename factory_table_type::map_type& factory_table =
            factory_table_type.get();

        typename component_id_table_type::map_type::iterator
            it = c_id_table.begin(), end = c_id_table.end();

        // this is the first request, so we use the type counter, and then
        // increment it.
        if (it == end)
            it = c_id_table.insert(key, type_counter++);

        // TODO: finish

    } // }}}

    prefixes_type resolve_id(component_id_type key)
    { // {{{ resolve_id implementation 
        typename database_mutex_type::scoped_lock l(mutex_);
        // TODO: implement
    } // }}}
    
    component_id_type resolve_name(component_name_type const& key)
    { // {{{ resolve_name implementation
        typename database_mutex_type::scoped_lock l(mutex_);
        // TODO: implement
    } // }}} 
    
    void unbind(component_name_type const& key)
    { // {{{ unbind implementation
        typename database_mutex_type::scoped_lock l(mutex_);
        // TODO: implement
    } // }}} 

    // {{{ action types
    enum actions
    {
        namespace_bind,
        namespace_resolve_name,
        namespace_resolve_id,
        namespace_unbind,
    };
  
    typedef hpx::actions::result_action2<
        component_namespace<Database>,
        /* return type */ component_id_type,
        /* enum value */  namespace_bind,
        /* arguments */   component_name_type const&, prefix_type,
        &component_namespace<Database>::bind
    > bind_action;
    
    typedef hpx::actions::result_action1<
        component_namespace<Database>,
        /* return type */ prefixes_type,
        /* enum value */  namespace_resolve_id,
        /* arguments */   component_id_type,
        &component_namespace<Database>::resolve_id
    > resolve_id_action;
    
    typedef hpx::actions::result_action1<
        component_namespace<Database>,
        /* return type */ component_id_type,
        /* enum value */  namespace_resolve_name,
        /* arguments */   component_name_type const&,
        &component_namespace<Database>::resolve_name
    > resolve_name_action;
    
    typedef hpx::actions::result_action1<
        component_namespace<Database>,
        /* return type */ void,
        /* enum value */  namespace_unbind,
        /* arguments */   component_name_type const&,
        &component_namespace<Database>::unbind
    > unbind_action;
    // }}}
};

}}}}

#endif // HPX_A16135FC_AA32_444F_BB46_549AD456A661

