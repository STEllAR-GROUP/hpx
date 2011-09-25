////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(HPX_A16135FC_AA32_444F_BB46_549AD456A661)
#define HPX_A16135FC_AA32_444F_BB46_549AD456A661

#include <set>

#include <boost/format.hpp>
#include <boost/assert.hpp>
#include <boost/utility/binary.hpp>

#include <hpx/hpx_fwd.hpp>
#include <hpx/exception.hpp>
#include <hpx/util/logging.hpp>
#include <hpx/util/insert_checked.hpp>
#include <hpx/runtime/agas/traits.hpp>
#include <hpx/runtime/agas/database/table.hpp>
#include <hpx/runtime/agas/namespace/response.hpp>
#include <hpx/runtime/components/component_type.hpp>
#include <hpx/runtime/components/server/fixed_component_base.hpp>

namespace hpx { namespace agas { namespace server
{

template <typename Database, typename Protocol>
struct component_namespace :
  components::fixed_component_base<
    HPX_AGAS_COMPONENT_NS_MSB, HPX_AGAS_COMPONENT_NS_LSB, // constant GID
    component_namespace<Database, Protocol>
  >
{
    // {{{ nested types
    typedef typename traits::database::mutex_type<Database>::type
        database_mutex_type;

    typedef std::string component_name_type;
    typedef int component_id_type;
    typedef boost::uint32_t prefix_type;

    typedef std::set<prefix_type> prefixes_type;

    typedef response<Protocol> response_type;

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
    component_namespace(
        std::string const& name = "root_component_namespace"
        )
      : mutex_()
      , component_ids_(std::string("hpx.agas.") + name + ".id")
      , factories_(std::string("hpx.agas.") + name + ".factory")
      , type_counter(components::component_first_dynamic)
    {
        traits::initialize_mutex(mutex_);
    }

    response_type bind_prefix(
        component_name_type const& key
      , naming::gid_type const& prefix
        )
    {
        return bind_prefix(key, naming::get_prefix_from_gid(prefix), throws);
    }

    response_type bind_prefix(
        component_name_type const& key
      , prefix_type prefix
        )
    {
        return bind_prefix(key, prefix, throws);
    }

    response_type bind_prefix(
        component_name_type const& key
      , naming::gid_type const& prefix
      , error_code& ec
        )
    {
        return bind_prefix(key, naming::get_prefix_from_gid(prefix), ec);
    }

    response_type bind_prefix(
        component_name_type const& key
      , prefix_type prefix
      , error_code& ec
        )
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
        {
            if (HPX_UNLIKELY(!util::insert_checked(c_id_table.insert(
                    std::make_pair(key, type_counter)), cit)))
            {
                HPX_THROWS_IF(ec, lock_error
                    , "component_namespace::bind_prefix"
                    , "component id table insertion failed due to a locking "
                      "error or memory corruption")
                return response_type();
            }

            // If the insertion succeeded, we need to increment the type
            // counter. 
            ++type_counter;
        }
 
        typename factory_table_type::map_type::iterator
            fit = factory_table.find(cit->second), fend = factory_table.end();

        if (fit == fend)
        {
            // Instead of creating a temporary and then inserting it, we insert
            // an empty set, then put the prefix into said set. This should
            // prevent a copy, though most compilers should be able to optimize
            // this without our help.
            if (HPX_UNLIKELY(!util::insert_checked(factory_table.insert(
                    std::make_pair(cit->second, prefixes_type())), fit)))
            {
                HPX_THROWS_IF(ec, lock_error
                    , "component_namespace::bind_prefix"
                    , "factory table insertion failed due to a locking "
                      "error or memory corruption")
                return response_type();
            }
        }

        fit->second.insert(prefix);

        LAGAS_(info) << (boost::format(
            "component_namespace::bind_prefix, key(%1%), prefix(%2%), "
            "ctype(%3%)")
            % key % prefix % cit->second);

        if (&ec != &throws)
            ec = make_success_code();

        return response_type(component_ns_bind_prefix, cit->second);
    } // }}}

    response_type bind_name(
        component_name_type const& key
        )
    {
        return bind_name(key, throws);
    }
    
    response_type bind_name(
        component_name_type const& key
      , error_code& ec
        )
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
        {
            if (HPX_UNLIKELY(!util::insert_checked(c_id_table.insert(
                    std::make_pair(key, type_counter)), it)))
            {
                HPX_THROWS_IF(ec, lock_error
                  , "component_namespace::bind_name"
                  , "component id table insertion failed due to a locking "
                    "error or memory corruption");
                return response_type();
            }

            // If the insertion succeeded, we need to increment the type
            // counter. 
            ++type_counter;
        }

        LAGAS_(info) << (boost::format(
            "component_namespace::bind_name, key(%1%), ctype(%2%)")
            % key % it->second);

        if (&ec != &throws)
            ec = make_success_code();

        return response_type(component_ns_bind_name, it->second);
    } // }}} 

    response_type resolve_id(
        components::component_type key
        )
    {
        return resolve_id(component_id_type(key), throws);
    }

    response_type resolve_id(
        component_id_type key
        )
    {
        return resolve_id(key, throws);
    }

    response_type resolve_id(
        components::component_type key
      , error_code& ec
        )
    {
        return resolve_id(component_id_type(key), ec);
    }

    response_type resolve_id(
        component_id_type key
      , error_code& ec
        )
    { // {{{ resolve_id implementation 
        typename database_mutex_type::scoped_lock l(mutex_);

        // Load the table.
        typename factory_table_type::map_type const& factory_table =
            factories_.get();
        
        typename factory_table_type::map_type::const_iterator
            it = factory_table.find(key), end = factory_table.end();

        // REVIEW: Should we differentiate between these two cases? Should we
        // throw an exception if it->second.empty()? It should be impossible.
        if (it == end || it->second.empty())
        {
            LAGAS_(info) << (boost::format(
                "component_namespace::resolve_id, key(%1%), localities(0)")
                % key);

            if (&ec != &throws)
                ec = make_success_code();

            return response_type(component_ns_resolve_id
                               , std::vector<boost::uint32_t>());
        }

        else
        {
            std::vector<boost::uint32_t> p;

            typename prefixes_type::const_iterator pit = it->second.begin()
                                                 , pend = it->second.end();

            for (; pit != pend; ++pit)
                p.push_back(*pit); 

            LAGAS_(info) << (boost::format(
                "component_namespace::resolve_id, key(%1%), localities(%2%)")
                % key % it->second.size());

            if (&ec != &throws)
                ec = make_success_code();

            return response_type(component_ns_resolve_id, p);
        } 
    } // }}}

    response_type resolve_name(
        component_name_type const& key
        )
    {
        return resolve_name(key, throws);
    }
    
    response_type resolve_name(
        component_name_type const& key
      , error_code& ec
        )
    { // {{{ resolve_name implementation
        typename database_mutex_type::scoped_lock l(mutex_);

        // Load the table.
        typename component_id_table_type::map_type& c_id_table =
            component_ids_.get();
        
        typename component_id_table_type::map_type::iterator
            it = c_id_table.find(key), end = c_id_table.end();

        if (it == end)
        {
            // REVIEW: Right response?
            LAGAS_(info) << (boost::format(
                "component_namespace::resolve_name, key(%1%), "
                "response(no_success)")
                % key);

            if (&ec != &throws)
                ec = make_success_code();

            return response_type(component_ns_resolve_name
                               , components::component_invalid
                               , no_success);
        }
 
        LAGAS_(info) << (boost::format(
            "component_namespace::resolve_name, key(%1%), ctype(%2%)")
            % key % it->second);

        if (&ec != &throws)
            ec = make_success_code();

        return response_type(component_ns_resolve_name, it->second);
    } // }}} 

    response_type unbind(
        component_name_type const& key
        )
    {
        return unbind(key, throws);
    }
    
    response_type unbind(
        component_name_type const& key
      , error_code& ec
        )
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
        {
            LAGAS_(info) << (boost::format(
                "component_namespace::unbind, key(%1%), response(no_success)")
                % key);

            if (&ec != &throws)
                ec = make_success_code();

           return response_type(component_ns_unbind, no_success);
        }

        // REVIEW: If there are no localities with this type, should we throw
        // an exception here?
        factory_table.erase(it->second);
        c_id_table.erase(it);

        LAGAS_(info) << (boost::format(
            "component_namespace::unbind, key(%1%)")
            % key);

        if (&ec != &throws)
            ec = make_success_code();

        return response_type(component_ns_unbind);
    } // }}} 

    enum actions
    { // {{{ action enum
        namespace_bind_prefix  = BOOST_BINARY_U(0100000),
        namespace_bind_name    = BOOST_BINARY_U(0100001),
        namespace_resolve_id   = BOOST_BINARY_U(0100010),
        namespace_resolve_name = BOOST_BINARY_U(0100011),
        namespace_unbind       = BOOST_BINARY_U(0100100)
    }; // }}}
    
    typedef hpx::actions::result_action2<
        component_namespace<Database, Protocol>,
        /* return type */ response_type,
        /* enum value */  namespace_bind_prefix,
        /* arguments */   component_name_type const&, prefix_type,
        &component_namespace<Database, Protocol>::bind_prefix
      , threads::thread_priority_critical
    > bind_prefix_action;
    
    typedef hpx::actions::result_action1<
        component_namespace<Database, Protocol>,
        /* return type */ response_type,
        /* enum value */  namespace_bind_name,
        /* arguments */   component_name_type const&,
        &component_namespace<Database, Protocol>::bind_name
      , threads::thread_priority_critical
    > bind_name_action;
    
    typedef hpx::actions::result_action1<
        component_namespace<Database, Protocol>,
        /* return type */ response_type,
        /* enum value */  namespace_resolve_id,
        /* arguments */   component_id_type,
        &component_namespace<Database, Protocol>::resolve_id
      , threads::thread_priority_critical
    > resolve_id_action;
    
    typedef hpx::actions::result_action1<
        component_namespace<Database, Protocol>,
        /* return type */ response_type,
        /* enum value */  namespace_resolve_name,
        /* arguments */   component_name_type const&,
        &component_namespace<Database, Protocol>::resolve_name
      , threads::thread_priority_critical
    > resolve_name_action;
    
    typedef hpx::actions::result_action1<
        component_namespace<Database, Protocol>,
        /* return type */ response_type,
        /* enum value */  namespace_unbind,
        /* arguments */   component_name_type const&,
        &component_namespace<Database, Protocol>::unbind
      , threads::thread_priority_critical
    > unbind_action;
};

}}}

#endif // HPX_A16135FC_AA32_444F_BB46_549AD456A661

