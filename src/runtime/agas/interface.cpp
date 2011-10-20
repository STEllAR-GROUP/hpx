////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Adelstein-Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#include <hpx/runtime/naming/resolver_client.hpp>
#include <hpx/runtime/agas/interface.hpp>

namespace hpx { namespace agas
{

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

    return false;
}

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

}}

