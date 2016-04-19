//  Copyright (c) 2013 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#include <hpx/components/security/manage_certificate_authority.hpp>

extern "C"
{
    namespace security = hpx::components::security;

    // Manage root_certificate_authority
    security::server::root_certificate_authority*
        new_root_certificate_authority(security::key_pair const & key_pair)
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
            security::key_pair const & key_pair)
    {
        return new security::server::subordinate_certificate_authority(
            key_pair);
    }

    void delete_subordinate_certificate_authority(
        security::server::subordinate_certificate_authority*
            certificate_authority)
    {
        delete certificate_authority;
    }

    // subordinate_certificate_authority helpers
    void subordinate_certificate_authority_get_certificate_signing_request(
        security::server::subordinate_certificate_authority*
            certificate_authority
      , security::signed_certificate_signing_request* signed_csr)
    {
        *signed_csr = certificate_authority->get_certificate_signing_request();
    }

    void subordinate_certificate_authority_set_certificate(
        security::server::subordinate_certificate_authority*
            certificate_authority
      , security::signed_certificate const & signed_certificate)
    {
        certificate_authority->set_certificate(signed_certificate);
    }

    // Helpers
    void certificate_authority_sign_certificate_signing_request(
        security::server::certificate_authority_base* certificate_authority
      , security::signed_certificate_signing_request const & signed_csr
      , security::signed_certificate* signed_certificate)
    {
        *signed_certificate =
            certificate_authority->sign_certificate_signing_request(
                signed_csr);
    }

    void certificate_authority_get_certificate(
        security::server::certificate_authority_base* certificate_authority
      , security::signed_certificate* certificate)
    {
        *certificate = certificate_authority->get_certificate();
    }

    void certificate_authority_get_gid(
        security::server::certificate_authority_base* certificate_authority
      , hpx::naming::gid_type* base_gid)
    {
        *base_gid = certificate_authority->get_base_gid();
    }

    void certificate_authority_is_valid(
        security::server::certificate_authority_base* certificate_authority
      , bool* valid)
    {
        *valid = certificate_authority->is_valid();
    }
}
