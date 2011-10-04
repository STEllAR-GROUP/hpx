////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#include <hpx/runtime/actions/continuation_impl.hpp>
#include <hpx/runtime/agas/server/component_namespace.hpp>

namespace hpx { namespace agas { namespace server
{

response component_namespace::bind_prefix(
    std::string const& key
  , prefix_type prefix
  , error_code& ec
    )
{ // {{{ bind_prefix implementation
    database_mutex_type::scoped_lock l(mutex_);

    component_id_table_type::iterator cit = component_ids_.find(key)
                                    , cend = component_ids_.end();

    // This is the first request, so we use the type counter, and then
    // increment it.
    if (cit == cend)
    {
        if (HPX_UNLIKELY(!util::insert_checked(component_ids_.insert(
                std::make_pair(key, type_counter)), cit)))
        {
            HPX_THROWS_IF(ec, lock_error
                , "component_namespace::bind_prefix"
                , "component id table insertion failed due to a locking "
                  "error or memory corruption")
            return response();
        }

        // If the insertion succeeded, we need to increment the type
        // counter. 
        ++type_counter;
    }

    factory_table_type::iterator fit = factories_.find(cit->second)
                               , fend = factories_.end();

    if (fit == fend)
    {
        // Instead of creating a temporary and then inserting it, we insert
        // an empty set, then put the prefix into said set. This should
        // prevent a copy, though most compilers should be able to optimize
        // this without our help.
        if (HPX_UNLIKELY(!util::insert_checked(factories_.insert(
                std::make_pair(cit->second, prefixes_type())), fit)))
        {
            HPX_THROWS_IF(ec, lock_error
                , "component_namespace::bind_prefix"
                , "factory table insertion failed due to a locking "
                  "error or memory corruption")
            return response();
        }
    }

    fit->second.insert(prefix);

    LAGAS_(info) << (boost::format(
        "component_namespace::bind_prefix, key(%1%), prefix(%2%), "
        "ctype(%3%)")
        % key % prefix % cit->second);

    if (&ec != &throws)
        ec = make_success_code();

    return response(component_ns_bind_prefix, cit->second);
} // }}}

response component_namespace::bind_name(
    std::string const& key
  , error_code& ec
    )
{ // {{{ bind_name implementation
    database_mutex_type::scoped_lock l(mutex_);

    component_id_table_type::iterator it = component_ids_.find(key)
                                    , end = component_ids_.end();

    // If the name is not in the table, register it (this is only done so
    // we can implement a backwards compatible get_component_id).
    if (it == end)
    {
        if (HPX_UNLIKELY(!util::insert_checked(component_ids_.insert(
                std::make_pair(key, type_counter)), it)))
        {
            HPX_THROWS_IF(ec, lock_error
              , "component_namespace::bind_name"
              , "component id table insertion failed due to a locking "
                "error or memory corruption");
            return response();
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

    return response(component_ns_bind_name, it->second);
} // }}} 

response component_namespace::resolve_id(
    component_id_type key
  , error_code& ec
    )
{ // {{{ resolve_id implementation 
    database_mutex_type::scoped_lock l(mutex_);

    factory_table_type::const_iterator it = factories_.find(key)
                                     , end = factories_.end();

    // REVIEW: Should we differentiate between these two cases? Should we
    // throw an exception if it->second.empty()? It should be impossible.
    if (it == end || it->second.empty())
    {
        LAGAS_(info) << (boost::format(
            "component_namespace::resolve_id, key(%1%), localities(0)")
            % key);

        if (&ec != &throws)
            ec = make_success_code();

        return response(component_ns_resolve_id
                           , std::vector<boost::uint32_t>());
    }

    else
    {
        std::vector<boost::uint32_t> p;

        prefixes_type::const_iterator pit = it->second.begin()
                                    , pend = it->second.end();

        for (; pit != pend; ++pit)
            p.push_back(*pit); 

        LAGAS_(info) << (boost::format(
            "component_namespace::resolve_id, key(%1%), localities(%2%)")
            % key % it->second.size());

        if (&ec != &throws)
            ec = make_success_code();

        return response(component_ns_resolve_id, p);
    } 
} // }}}

response component_namespace::resolve_name(
    std::string const& key
  , error_code& ec
    )
{ // {{{ resolve_name implementation
    database_mutex_type::scoped_lock l(mutex_);

    component_id_table_type::iterator it = component_ids_.find(key)
                                    , end = component_ids_.end();

    if (it == end)
    {
        // REVIEW: Right response?
        LAGAS_(info) << (boost::format(
            "component_namespace::resolve_name, key(%1%), "
            "response(no_success)")
            % key);

        if (&ec != &throws)
            ec = make_success_code();

        return response(component_ns_resolve_name
                           , components::component_invalid
                           , no_success);
    }

    LAGAS_(info) << (boost::format(
        "component_namespace::resolve_name, key(%1%), ctype(%2%)")
        % key % it->second);

    if (&ec != &throws)
        ec = make_success_code();

    return response(component_ns_resolve_name, it->second);
} // }}} 

response component_namespace::unbind(
    std::string const& key
  , error_code& ec
    )
{ // {{{ unbind implementation
    database_mutex_type::scoped_lock l(mutex_);
    
    component_id_table_type::iterator it = component_ids_.find(key)
                                    , end = component_ids_.end();

    // REVIEW: Should this be an error?
    if (it == end)
    {
        LAGAS_(info) << (boost::format(
            "component_namespace::unbind, key(%1%), response(no_success)")
            % key);

        if (&ec != &throws)
            ec = make_success_code();

       return response(component_ns_unbind, no_success);
    }

    // REVIEW: If there are no localities with this type, should we throw
    // an exception here?
    factories_.erase(it->second);
    component_ids_.erase(it);

    LAGAS_(info) << (boost::format(
        "component_namespace::unbind, key(%1%)")
        % key);

    if (&ec != &throws)
        ec = make_success_code();

    return response(component_ns_unbind);
} // }}} 

}}}

