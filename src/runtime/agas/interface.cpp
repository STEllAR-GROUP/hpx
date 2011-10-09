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

    if (agas_.registerid(name, new_gid, ec) && ec)
        return true;

    return false;
}

bool unregister_name(
    std::string const& name
  , error_code& ec
    )
{
    naming::resolver_client& agas_ = naming::get_agas_client();

    naming::gid_type gid;

    if (agas_.unregisterid(name, gid, ec) && ec)
    {
        // Let the GID go out of scope
        naming::id_type id(gid, naming::id_type::managed);
        return true;
    }

    return false;
}

bool query_name(
    std::string const& ns_name
  , naming::id_type& gid
  , error_code& ec
    )
{
    naming::resolver_client& agas_ = naming::get_agas_client();

    if (agas_.queryid(name, gid.get_gid(), ec) && ec)
        return true;

    return false;
}

bool query_name(
    std::string const& ns_name
  , naming::gid_type& gid
  , error_code& ec
    )
{
    naming::resolver_client& agas_ = naming::get_agas_client();

    if (agas_.queryid(name, gid.get_gid(), ec) && ec)
        return true;

    return false;
}

}}

