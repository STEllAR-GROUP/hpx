////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(HPX_A16135FC_AA32_444F_BB46_549AD456A661)
#define HPX_A16135FC_AA32_444F_BB46_549AD456A661

#include <vector>

#include <boost/assert.hpp>

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/agas/traits.hpp>
#include <hpx/runtime/agas/database/table.hpp>
#include <hpx/runtime/components/component_type.hpp>
#include <hpx/runtime/components/server/fixed_component_base.hpp>

namespace hpx { namespace agas { namespace server
{

template <typename Database>
struct HPX_COMPONENT_EXPORT component_namespace :
  components::fixed_component_base<
    HPX_AGAS_COMPONENT_NS_MSB, HPX_AGAS_COMPONENT_NS_LSB, // constant GID
    component_namespace<Database>
  >
{
    // {{{ nested types
    typedef typename traits::database::mutex_type<Database>::type
        database_mutex_type;

    typedef std::string component_name_type;
    typedef int component_id_type;
    typedef boost::uint32_t prefix_type;

    // I want this to be a boost::unordered_set<>, but for backwards
    // compatibility, it's an std::vector<>
    typedef std::vector<prefix_type> prefixes_type;

    typedef table<Database, component_name_type, component_id_type>
        component_id_table_type; 

    typedef table<Database, component_id_type, prefixes_type>
        factory_table_type;
    // }}}

  private:
    database_mutex_type mutex_;
    component_id_table_type component_ids_;
    factory_table_type factories_;
    component_id_type type_counter; 
 
  public:
    component_namespace(std::string const& name
                                = "component_namespace")
      : mutex_(),
        component_ids_(std::string("hpx.agas.") + name + ".id"),
        factories_(std::string("hpx.agas.") + name + ".factory"),
        type_counter(components::component_first_dynamic)
    { traits::initialize_mutex(mutex_); }

    component_id_type
    bind_prefix(component_name_type const& key, prefix_type prefix)
    { // {{{ bind_prefix implementation
        typename database_mutex_type::scoped_lock l(mutex_);

        // Always load the table once, as the operation might be expensive for
        // some backends and the compiler may not be able to optimize this away
        // if the database backend has implicit/builtin atomic semantics.
        typename component_id_table_type::map_type& c_id_table =
            component_ids_.get();
        
        typename factory_table_type::map_type& factory_table =
            factories_.get();

        typename component_id_table_type::map_type::iterator
            cit = c_id_table.find(key), cend = c_id_table.end();

        // This is the first request, so we use the type counter, and then
        // increment it.
        if (cit == cend)
            cit = c_id_table.insert(typename
                component_id_table_type::map_type::value_type
                    (key, type_counter++)).first;

        // REVIEW: make this a locking exception?
        BOOST_ASSERT(cit != cend);
        
        typename factory_table_type::map_type::iterator
            fit = factory_table.find(cit->second), fend = factory_table.end();

        if (fit == fend)
            // Instead of creating a temporary and then inserting it, we insert
            // an empty set, then put the prefix into said set. This should
            // prevent a copy, though most compilers should be able to optimize
            // this without our help.
            fit = factory_table.insert(typename
                factory_table_type::map_type::value_type
                    (cit->second, prefixes_type())).first;

        // REVIEW: make this a locking exception?
        BOOST_ASSERT(fit != fend);

        fit->second.push_back(prefix);

        return cit->second;
    } // }}}
    
    component_id_type bind_name(component_name_type const& key) 
    { // {{{ bind_name implementation
        typename database_mutex_type::scoped_lock l(mutex_);

        // Load the table.
        typename component_id_table_type::map_type& c_id_table =
            component_ids_.get();
        
        typename component_id_table_type::map_type::iterator
            it = c_id_table.find(key), end = c_id_table.end();

        // If the name is not in the table, register it (this is only done so
        // we can implement a backwards compatible get_component_id).
        if (it == end)
            it = c_id_table.insert(typename
                component_id_table_type::map_type::value_type
                    (key, type_counter++)).first;

        // REVIEW: make this a locking exception?
        BOOST_ASSERT(it != end);

        return it->second;
    } // }}} 

    prefixes_type resolve_id(component_id_type key) 
    { // {{{ resolve_id implementation 
        typename database_mutex_type::scoped_lock l(mutex_);

        // Load the table.
        typename factory_table_type::map_type const& factory_table =
            factories_.get();
        
        typename factory_table_type::map_type::const_iterator
            it = factory_table.find(key), end = factory_table.end();

        if (it == end)
            return prefixes_type();

        return it->second;
    } // }}}
    
    component_id_type resolve_name(component_name_type const& key) 
    { // {{{ resolve_name implementation
        typename database_mutex_type::scoped_lock l(mutex_);

        // Load the table.
        typename component_id_table_type::map_type& c_id_table =
            component_ids_.get();
        
        typename component_id_table_type::map_type::iterator
            it = c_id_table.find(key), end = c_id_table.end();

        // If the name is not in the table, register it (this is only done so
        // we can implement a backwards compatible get_component_id).
        if (it == end)
            return components::component_invalid;

        return it->second;
    } // }}} 
    
    bool unbind(component_name_type const& key)
    { // {{{ unbind implementation
        typename database_mutex_type::scoped_lock l(mutex_);

        typename component_id_table_type::map_type& c_id_table =
            component_ids_.get();
        
        typename factory_table_type::map_type& factory_table =
            factories_.get();

        typename component_id_table_type::map_type::iterator
            it = c_id_table.find(key), end = c_id_table.end();

        // REVIEW: Should this be an error?
        if (it == end)
          return false;

        // REVIEW: If there are no localities with this type, should we throw
        // an exception here?
        factory_table.erase(it->second);
        c_id_table.erase(it);

        return true;
    } // }}} 

    enum actions
    { // {{{ action enum
        namespace_bind_prefix,
        namespace_bind_name,
        namespace_resolve_name,
        namespace_resolve_id,
        namespace_unbind
    }; // }}}
    
    typedef hpx::actions::result_action2<
        component_namespace<Database>,
        /* return type */ component_id_type,
        /* enum value */  namespace_bind_prefix,
        /* arguments */   component_name_type const&, prefix_type,
        &component_namespace<Database>::bind_prefix
    > bind_prefix_action;
    
    typedef hpx::actions::result_action1<
        component_namespace<Database>,
        /* return type */ component_id_type,
        /* enum value */  namespace_resolve_name,
        /* arguments */   component_name_type const&,
        &component_namespace<Database>::bind_name
    > bind_name_action;
    
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
        /* return type */ bool,
        /* enum value */  namespace_unbind,
        /* arguments */   component_name_type const&,
        &component_namespace<Database>::unbind
    > unbind_action;
};

}}}

#endif // HPX_A16135FC_AA32_444F_BB46_549AD456A661

