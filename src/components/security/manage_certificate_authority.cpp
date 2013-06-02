//  Copyright (c) 2013 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_fwd.hpp>
#include <hpx/components/security/manage_certificate_authority.hpp>

extern "C"
{
    namespace security = hpx::components::security;

    // Manage root_certificate_authority
    security::server::root_certificate_authority*
        new_root_certificate_authority(security::server::key_pair const & key_pair)
    {
        return new security::server::root_certificate_authority(key_pair);
    }

    void delete_root_certificate_authority(
        security::server::root_certificate_authority* certificate_authority)
    {
        delete certificate_authority;
    }

    // Manage subordinate_certificate_authority
    security::server::subordinate_certificate_authority*
        new_subordinate_certificate_authority(
            security::server::key_pair const & key_pair
          , hpx::naming::id_type const & gid)
    {
        return new security::server::subordinate_certificate_authority(
            key_pair, gid);
    }

    void delete_subordinate_certificate_authority(
        security::server::subordinate_certificate_authority*
            certificate_authority)
    {
        delete certificate_authority;
    }

    // Helpers
    void certificate_authority_get_certificate(
        security::server::certificate_authority_base* certificate_authority
      , security::server::signed_type<
            security::server::certificate
        >* certificate)
    {
        *certificate = certificate_authority->get_certificate();
    }

    void ca_get_gid(
        security::server::certificate_authority_base* certificate_authority
      , hpx::naming::gid_type* gid)
    {
        *gid = certificate_authority->get_base_gid();
    }
}

