////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Adelstein-Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#include <hpx/runtime/naming/resolver_client.hpp>
#include <hpx/runtime/agas/interface.hpp>
#include <hpx/runtime/actions/continuation.hpp>
#include <hpx/runtime/components/stubs/runtime_support.hpp>

namespace hpx { namespace agas
{
///////////////////////////////////////////////////////////////////////////////
bool register_name(
    std::string const& name
  , naming::id_type const& gid
  , error_code& ec
    )
{
    naming::resolver_client& agas_ = naming::get_agas_client();

    // We need to modify the reference count.
    naming::gid_type& mutable_gid = const_cast<naming::id_type&>(gid).get_gid();

    naming::gid_type new_gid;

    if (naming::get_credit_from_gid(mutable_gid) != 0)
    {
        new_gid = split_credits_for_gid(mutable_gid);

        // Credit exhaustion - we need to get more.
        if (0 == naming::get_credit_from_gid(new_gid))
        {
            BOOST_ASSERT(1 == naming::get_credit_from_gid(mutable_gid));
            agas_.incref(new_gid, 2 * HPX_INITIAL_GLOBALCREDIT);

            naming::add_credit_to_gid(new_gid, HPX_INITIAL_GLOBALCREDIT);
            naming::add_credit_to_gid(mutable_gid, HPX_INITIAL_GLOBALCREDIT);
        }
    }

    else
        new_gid = mutable_gid;

    if (agas_.register_name(name, new_gid, ec) && !ec)
        return true;

    // Return the credit to the GID, as the operation failed
    if (naming::get_credit_from_gid(mutable_gid) != 0)
    {
        naming::add_credit_to_gid(mutable_gid,
            naming::get_credit_from_gid(new_gid));
    }

    return false;
}

lcos::future<bool, response> register_name_async(
    std::string const& name
  , naming::id_type const& id
    )
{
    // We need to modify the reference count.
    naming::gid_type& mutable_gid = const_cast<naming::id_type&>(id).get_gid();
    naming::gid_type new_gid;

    // FIXME: combine incref with register_name, if needed
    if (naming::get_credit_from_gid(mutable_gid) != 0)
    {
        new_gid = split_credits_for_gid(mutable_gid);

        // Credit exhaustion - we need to get more.
        if (0 == naming::get_credit_from_gid(new_gid))
        {
            BOOST_ASSERT(1 == naming::get_credit_from_gid(mutable_gid));
            naming::get_agas_client().incref(new_gid, 2 * HPX_INITIAL_GLOBALCREDIT);

            naming::add_credit_to_gid(new_gid, HPX_INITIAL_GLOBALCREDIT);
            naming::add_credit_to_gid(mutable_gid, HPX_INITIAL_GLOBALCREDIT);
        }
    }
    else
        new_gid = mutable_gid;

    request req(symbol_ns_bind, name, new_gid);

    naming::id_type const target = bootstrap_symbol_namespace_id();

    return stubs::symbol_namespace::service_async<bool>(target, req);
}

///////////////////////////////////////////////////////////////////////////////
bool unregister_name(
    std::string const& name
  , error_code& ec
    )
{
    naming::resolver_client& agas_ = naming::get_agas_client();

    naming::gid_type raw_gid;

    if (agas_.unregister_name(name, raw_gid, ec) && !ec)
    {
        // If the GID has a reference count, return it to AGAS.
        if (naming::get_credit_from_gid(raw_gid) != 0)
            // When this id_type goes out of scope, it's deleter will
            // take care of the reference count.
            naming::id_type gid(raw_gid, naming::id_type::managed);

        return true;
    }

    return false;
}

bool unregister_name(
    std::string const& name
  , naming::id_type& gid
  , error_code& ec
    )
{
    naming::resolver_client& agas_ = naming::get_agas_client();

    naming::gid_type raw_gid;

    if (agas_.unregister_name(name, raw_gid, ec) && !ec)
    {
        if (naming::get_credit_from_gid(raw_gid) != 0)
            gid = naming::id_type(raw_gid, naming::id_type::managed);
        else
            gid = naming::id_type(raw_gid, naming::id_type::unmanaged);

        return true;
    }

    return false;
}

lcos::future<naming::id_type, response> unregister_name_async(
    std::string const& name
    )
{
    request req(symbol_ns_unbind, name);

    naming::id_type const target = bootstrap_symbol_namespace_id();

    return stubs::symbol_namespace::service_async<naming::id_type>(target, req);
}

///////////////////////////////////////////////////////////////////////////////
bool resolve_name(
    std::string const& name
  , naming::id_type& gid
  , error_code& ec
    )
{
    naming::resolver_client& agas_ = naming::get_agas_client();

    naming::gid_type raw_gid;

    if (agas_.resolve_name(name, raw_gid, ec) && !ec)
    {
        if (naming::get_credit_from_gid(raw_gid) != 0)
            gid = naming::id_type(raw_gid, naming::id_type::managed);
        else
            gid = naming::id_type(raw_gid, naming::id_type::unmanaged);

        return true;
    }

    return false;
}

bool resolve_name(
    std::string const& name
  , naming::gid_type& gid
  , error_code& ec
    )
{
    naming::resolver_client& agas_ = naming::get_agas_client();

    if (agas_.resolve_name(name, gid, ec) && !ec)
        return true;

    return false;
}

lcos::future<naming::id_type, response> resolve_name_async(
    std::string const& name
    )
{
    request req(symbol_ns_resolve, name);

    naming::id_type const gid = bootstrap_symbol_namespace_id();

    return stubs::symbol_namespace::service_async<naming::id_type>(gid, req);
}

bool is_local_address(
    naming::gid_type const& gid
  , error_code& ec
    )
{
    return naming::get_agas_client().is_local_address(gid, ec);
}

bool is_local_address(
    naming::gid_type const& gid
  , naming::address& addr
  , error_code& ec
    )
{
    return naming::get_agas_client().is_local_address(gid, addr, ec);
}

bool is_local_address_cached(
    naming::gid_type const& gid
  , error_code& ec
    )
{
    return naming::get_agas_client().is_local_address_cached(gid, ec);
}

bool is_local_address_cached(
    naming::gid_type const& gid
  , naming::address& addr
  , error_code& ec
    )
{
    return naming::get_agas_client().is_local_address_cached(gid, addr, ec);
}

bool is_local_lva_encoded_address(
    naming::gid_type const& gid
    )
{
    return naming::get_agas_client().is_local_lva_encoded_address(gid);
}

void garbage_collect_non_blocking(
    error_code& ec
    )
{
    naming::get_agas_client().garbage_collect_non_blocking(ec);
}

void garbage_collect(
    error_code& ec
    )
{
    naming::get_agas_client().garbage_collect(ec);
}

/// \brief Invoke an asynchronous garbage collection step on the given target
///        locality.
void garbage_collect_non_blocking(
    naming::id_type const& id
  , error_code& ec
    )
{
    try {
        components::stubs::runtime_support::garbage_collect_non_blocking(id);
    }
    catch (hpx::exception const& e) {
        if (&ec == &throws)
            throw;
        ec = make_error_code(e.get_error(), e.what());
    }
}

/// \brief Invoke a synchronous garbage collection step on the given target
///        locality.
void garbage_collect(
    naming::id_type const& id
  , error_code& ec
    )
{
    try {
        components::stubs::runtime_support::garbage_collect(id);
    }
    catch (hpx::exception const& e) {
        if (&ec == &throws)
            throw;
        ec = make_error_code(e.get_error(), e.what());
    }
}

/// \brief Return an id_type referring to the console locality.
naming::id_type get_console_locality(
    error_code& ec
    )
{
    naming::gid_type console;
    naming::get_agas_client().get_console_locality(console, ec);
    if (ec) return naming::invalid_id;

    return naming::id_type(console, naming::id_type::unmanaged);
}
}}

