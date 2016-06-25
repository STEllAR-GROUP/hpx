//  Copyright (c) 2007-2012 Hartmut Kaiser
//  Copyright (c) 2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#include <hpx/runtime/naming/name.hpp>
#include <hpx/runtime/naming/address.hpp>
#include <hpx/runtime/naming/resolver_client.hpp>
#include <hpx/runtime/applier/applier.hpp>
#include <hpx/runtime/applier/bind_naming_wrappers.hpp>

namespace hpx { namespace applier
{
    // helper functions allowing to bind and unbind a GID to a given address
    // without having to directly refer to the resolver_client
    bool bind_gid_local(naming::gid_type const& gid_, naming::address const& addr,
        error_code& ec)
    {
        applier* appl = get_applier_ptr();
        if (nullptr == appl) {
            HPX_THROWS_IF(ec, invalid_status, "applier::bind_gid_local",
                "applier is not valid");
            return false;
        }
        return appl->get_agas_client().bind_local(gid_, addr, ec);
    }

    void unbind_gid_local(naming::gid_type const& gid_, error_code& ec)
    {
        if (gid_) {
            applier* appl = get_applier_ptr();
            if (nullptr == appl) {
                HPX_THROWS_IF(ec, invalid_status, "applier::unbind_gid_local",
                    "applier is not valid");
            }
            else {
                appl->get_agas_client().unbind_local(gid_, ec);
            }
        }
        else {
            HPX_THROWS_IF(ec, bad_parameter, "applier::unbind_gid",
                "cannot dereference invalid GID");
        }
    }

    bool bind_range_local(naming::gid_type const& gid, std::size_t count,
        naming::address const& addr, std::size_t offset, error_code& ec)
    {
        applier* appl = get_applier_ptr();
        if (nullptr == appl) {
            HPX_THROWS_IF(ec, invalid_status, "applier::bind_range_local",
                "applier is not valid");
            return false;
        }
        return appl->get_agas_client().bind_range_local(gid, count, addr, offset, ec);
    }

    void unbind_range_local(naming::gid_type const& gid, std::size_t count,
        error_code& ec)
    {
        applier* appl = get_applier_ptr();
        if (nullptr == appl) {
            HPX_THROWS_IF(ec, invalid_status, "applier::unbind_range_local",
                "applier is not valid");
        }
        else {
            appl->get_agas_client().unbind_range_local(gid, count, ec);
        }
    }
}}

