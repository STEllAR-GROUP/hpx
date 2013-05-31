//  Copyright (c) 2013 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_fwd.hpp>
#include <hpx/components/security/manage_ca.hpp>

extern "C"
{
    // manage root-CA
    hpx::components::security::server::root_certificate_authority*
        create_root_ca(hpx::components::security::server::key_pair const& k)
    {
        return new hpx::components::security::server::root_certificate_authority(k);
    }

    void delete_root_ca(
        hpx::components::security::server::root_certificate_authority* ca)
    {
        delete ca;
    }

    // manage sub-CA
    hpx::components::security::server::subordinate_certificate_authority*
        create_sub_ca(
            hpx::components::security::server::key_pair const& k
          , hpx::naming::id_type const& root_ca_gid)
    {
        return new hpx::components::security::server::subordinate_certificate_authority(
            k, root_ca_gid);
    }

    void delete_sub_ca(
        hpx::components::security::server::subordinate_certificate_authority* ca)
    {
        delete ca;
    }

    // common helpers
    void ca_get_certificate(
        hpx::components::security::server::certificate_authority_base* ca
      , hpx::components::security::server::signed_type<
            hpx::components::security::server::certificate
        >* cert)
    {
        *cert = ca->get_certificate();
    }

    void ca_get_gid(
        hpx::components::security::server::certificate_authority_base* ca
      , hpx::naming::gid_type* gid)
    {
        *gid = ca->get_base_gid();
    }
}

