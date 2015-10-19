//  Copyright (c) 2013 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_COMPONENTS_SECURITY_MANAGE_CA_MAY_30_2013_0720PM)
#define HPX_COMPONENTS_SECURITY_MANAGE_CA_MAY_30_2013_0720PM

#include <hpx/config.hpp>

#if defined(HPX_HAVE_SECURITY)

#include <hpx/runtime/naming/name.hpp>

#include <hpx/components/security/key_pair.hpp>
#include <hpx/components/security/server/root_certificate_authority.hpp>
#include <hpx/components/security/server/subordinate_certificate_authority.hpp>

extern "C"
{
    namespace security = hpx::components::security;

    // Manage root_certificate_authority
    HPX_COMPONENT_EXPORT
        security::server::root_certificate_authority*
            new_root_certificate_authority(security::key_pair const &);
    HPX_COMPONENT_EXPORT
        void delete_root_certificate_authority(
            security::server::root_certificate_authority*);

    // Manage subordinate_certificate_authority
    HPX_COMPONENT_EXPORT
        security::server::subordinate_certificate_authority*
            new_subordinate_certificate_authority(
                security::key_pair const &);
    HPX_COMPONENT_EXPORT
        void delete_subordinate_certificate_authority(
            security::server::subordinate_certificate_authority*);

    // subordinate_certificate_authority helpers
    HPX_COMPONENT_EXPORT
        void subordinate_certificate_authority_get_certificate_signing_request(
            security::server::subordinate_certificate_authority*
          , security::signed_certificate_signing_request*);

    HPX_COMPONENT_EXPORT
        void subordinate_certificate_authority_set_certificate(
            security::server::subordinate_certificate_authority*
          , security::signed_certificate const &);

    // Helpers
    HPX_COMPONENT_EXPORT
        void certificate_authority_sign_certificate_signing_request(
            security::server::certificate_authority_base*
          , security::signed_certificate_signing_request const &
          , security::signed_certificate*);

    HPX_COMPONENT_EXPORT
        void certificate_authority_get_certificate(
            security::server::certificate_authority_base*
          , security::signed_certificate*);

    HPX_COMPONENT_EXPORT
        void certificate_authority_get_gid(
            security::server::certificate_authority_base*
          , hpx::naming::gid_type*);

    HPX_COMPONENT_EXPORT
        void certificate_authority_is_valid(
            security::server::certificate_authority_base*
          , bool*);
}

#endif

#endif
