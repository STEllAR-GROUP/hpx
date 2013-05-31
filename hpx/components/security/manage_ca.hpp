//  Copyright (c) 2013 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_COMPONENTS_SECURITY_MANAGE_CA_MAY_30_2013_0720PM)
#define HPX_COMPONENTS_SECURITY_MANAGE_CA_MAY_30_2013_0720PM

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/naming/name.hpp>

#include <hpx/components/security/server/key_pair.hpp>
#include <hpx/components/security/server/root_certificate_authority.hpp>
#include <hpx/components/security/server/subordinate_certificate_authority.hpp>

extern "C"
{
    // manage root-CA
    HPX_COMPONENT_EXPORT
        hpx::components::security::server::root_certificate_authority*
            create_root_ca(hpx::components::security::server::key_pair const&);
    HPX_COMPONENT_EXPORT
        void delete_root_ca(
            hpx::components::security::server::root_certificate_authority*);

    // manage sub-CA
    HPX_COMPONENT_EXPORT
        hpx::components::security::server::subordinate_certificate_authority*
            create_sub_ca(
                hpx::components::security::server::key_pair const&
              , hpx::naming::id_type const&);
    HPX_COMPONENT_EXPORT
        void delete_sub_ca(
            hpx::components::security::server::subordinate_certificate_authority*);

    // common helpers
    HPX_COMPONENT_EXPORT
        void ca_get_certificate(
            hpx::components::security::server::certificate_authority_base*
          , hpx::components::security::server::signed_type<
                hpx::components::security::server::certificate
            >*);

    HPX_COMPONENT_EXPORT
        void ca_get_gid(
            hpx::components::security::server::certificate_authority_base*
          , hpx::naming::gid_type*);
}

#endif

