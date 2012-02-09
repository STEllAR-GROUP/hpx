////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Adelstein-Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#include <hpx/runtime/naming/resolver_client.hpp>
#include <hpx/runtime/actions/continuation.hpp>
#include <hpx/runtime/agas/server/symbol_namespace.hpp>

namespace hpx { namespace agas
{

naming::gid_type bootstrap_symbol_namespace_gid()
{
    return naming::gid_type
        (HPX_AGAS_SYMBOL_NS_MSB, HPX_AGAS_SYMBOL_NS_LSB);
}

naming::id_type bootstrap_symbol_namespace_id()
{
    return naming::id_type
        (bootstrap_symbol_namespace_gid(), naming::id_type::unmanaged);
}

namespace server
{

// TODO: This isn't scalable, we have to update it every time we add a new 
// AGAS request/response type.
response symbol_namespace::service(
    request const& req
  , error_code& ec
    )
{ // {{{
    switch (req.get_action_code())
    {
        case symbol_ns_bind:
            return bind(req, ec);
        case symbol_ns_resolve:
            return resolve(req, ec);
        case symbol_ns_unbind:
            return unbind(req, ec);
        case symbol_ns_iterate_names:
            return iterate(req, ec);

        case primary_ns_allocate:
        case primary_ns_bind_gid:
        case primary_ns_resolve_gid:
        case primary_ns_free:
        case primary_ns_unbind_gid:
        case primary_ns_change_credit_non_blocking:
        case primary_ns_change_credit_sync:
        case primary_ns_localities:
        {
            LAGAS_(warning) <<
                "component_namespace::service, redirecting request to "
                "primary_namespace";
            return naming::get_agas_client().service(req, ec);
        }

        case component_ns_bind_prefix:
        case component_ns_bind_name:
        case component_ns_resolve_id:
        case component_ns_unbind:
        case component_ns_iterate_types:
        {
            LAGAS_(warning) <<
                "component_namespace::service, redirecting request to "
                "component_namespace";
            return naming::get_agas_client().service(req, ec);
        }

        default:
        case component_ns_service:
        case primary_ns_service:
        case symbol_ns_service:
        case invalid_request:
        {
            HPX_THROWS_IF(ec, bad_action_code
              , "component_namespace::service"
              , boost::str(boost::format(
                    "invalid action code encountered in request, "
                    "action_code(%x)")
                    % boost::uint16_t(req.get_action_code())));
            return response();
        }
    };
} // }}}

// TODO: do/undo semantics (e.g. transactions)
std::vector<response> symbol_namespace::bulk_service(
    std::vector<request> const& reqs
  , error_code& ec
    )
{
    std::vector<response> r;
    r.reserve(reqs.size());

    BOOST_FOREACH(request const& req, reqs)
    {
        error_code ign;
        r.push_back(service(req, ign));
    }

    return r;
}

response symbol_namespace::bind(
    request const& req
  , error_code& ec
    )
{ // {{{ bind implementation
    // parameters
    std::string key = req.get_name();
    naming::gid_type gid = req.get_gid();

    mutex_type::scoped_lock l(mutex_);

    gid_table_type::iterator it = gids_.find(key)
                           , end = gids_.end();

    if (it != end)
    {
        boost::uint16_t const credits = naming::get_credit_from_gid(gid);
        naming::gid_type raw_gid = it->second;
        naming::strip_credit_from_gid(raw_gid);
        naming::strip_credit_from_gid(gid);

        // increase reference count
        if (raw_gid == gid)
        {
            LAGAS_(info) << (boost::format(
                "symbol_namespace::bind, key(%1%), gid(%2%), old_credit(%3%), "
                "new_credit(%4%)")
                % key % gid
                % naming::get_credit_from_gid(it->second)
                % (naming::get_credit_from_gid(it->second) + credits));

            naming::add_credit_to_gid(it->second, credits);

            if (&ec != &throws)
                ec = make_success_code();

            return response(symbol_ns_bind);
        }

        naming::add_credit_to_gid(gid, credits);
        LAGAS_(info) << (boost::format(
            "symbol_namespace::bind, key(%1%), gid(%2%), response(no_success)")
            % key % gid);

        if (&ec != &throws)
            ec = make_success_code();

        return response(symbol_ns_bind, no_success);
    }

    if (HPX_UNLIKELY(!util::insert_checked(gids_.insert(
            std::make_pair(key, gid)))))
    {
        HPX_THROWS_IF(ec, lock_error
          , "symbol_namespace::bind"
          , "GID table insertion failed due to a locking error or "
            "memory corruption");
        return response();
    }

    LAGAS_(info) << (boost::format(
        "symbol_namespace::bind, key(%1%), gid(%2%)")
        % key % gid);

    if (&ec != &throws)
        ec = make_success_code();

    return response(symbol_ns_bind);
} // }}}

response symbol_namespace::resolve(
    request const& req
  , error_code& ec
    )
{ // {{{ resolve implementation
    // parameters
    std::string key = req.get_name();

    mutex_type::scoped_lock l(mutex_);

    gid_table_type::iterator it = gids_.find(key)
                           , end = gids_.end();

    if (it == end)
    {
        LAGAS_(info) << (boost::format(
            "symbol_namespace::resolve, key(%1%), response(no_success)")
            % key);

        if (&ec != &throws)
            ec = make_success_code();

        return response(symbol_ns_resolve
                      , naming::invalid_gid
                      , no_success);
    }

    if (&ec != &throws)
        ec = make_success_code();

    naming::gid_type gid;

    // Is this entry reference counted?
    if (naming::get_credit_from_gid(it->second) != 0)
    {
        gid = naming::split_credits_for_gid(it->second);

        LAGAS_(debug) << (boost::format(
            "symbol_namespace::resolve, split credits for entry, "
            "key(%1%), entry(%2%), gid(%3%)")
            % key % it->second % gid);

        // Credit exhaustion - we need to get more.
        if (0 == naming::get_credit_from_gid(gid))
        {
            BOOST_ASSERT(1 == naming::get_credit_from_gid(it->second));
            naming::get_agas_client().incref(gid, 2 * HPX_INITIAL_GLOBALCREDIT);

            naming::add_credit_to_gid(gid, HPX_INITIAL_GLOBALCREDIT);
            naming::add_credit_to_gid(it->second, HPX_INITIAL_GLOBALCREDIT);

            LAGAS_(debug) << (boost::format(
                "symbol_namespace::resolve, incremented entry credits, "
                "key(%1%), entry(%2%), gid(%3%)")
                % key % it->second % gid);
        }
    }

    else
        gid = it->second;

    LAGAS_(info) << (boost::format(
        "symbol_namespace::resolve, key(%1%), gid(%2%)")
        % key % gid);

    return response(symbol_ns_resolve, gid);
} // }}}

response symbol_namespace::unbind(
    request const& req
  , error_code& ec
    )
{ // {{{ unbind implementation
    // parameters
    std::string key = req.get_name();

    mutex_type::scoped_lock l(mutex_);

    gid_table_type::iterator it = gids_.find(key)
                           , end = gids_.end();

    if (it == end)
    {
        LAGAS_(info) << (boost::format(
            "symbol_namespace::unbind, key(%1%), response(no_success)")
            % key);

        if (&ec != &throws)
            ec = make_success_code();

        return response(symbol_ns_unbind, naming::invalid_gid, no_success);
    }

    const naming::gid_type gid = it->second;

    gids_.erase(it);

    LAGAS_(info) << (boost::format(
        "symbol_namespace::unbind, key(%1%), gid(%2%)")
        % key % gid);

    if (&ec != &throws)
        ec = make_success_code();

    return response(symbol_ns_unbind, gid);
} // }}}

// TODO: catch exceptions
response symbol_namespace::iterate(
    request const& req
  , error_code& ec
    )
{ // {{{ iterate implementation
    iterate_names_function_type f = req.get_iterate_names_function();

    mutex_type::scoped_lock l(mutex_);

    for (gid_table_type::iterator it = gids_.begin()
                                , end = gids_.end();
         it != end; ++it)
    {
        f(it->first, it->second);
    }

    LAGAS_(info) << "symbol_namespace::iterate";

    if (&ec != &throws)
        ec = make_success_code();

    return response(symbol_ns_iterate_names);
} // }}}

}}}

