//  Copyright (c) 2013 Jeroen Habraken
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_COMPONENTS_SECURITY_SERVER_CERTIFICATE_AUTHORITY_BASE_HPP
#define HPX_COMPONENTS_SECURITY_SERVER_CERTIFICATE_AUTHORITY_BASE_HPP

#include <hpx/config.hpp>

#if defined(HPX_HAVE_SECURITY)

#include <hpx/components/security/certificate.hpp>
#include <hpx/components/security/certificate_signing_request.hpp>
#include <hpx/components/security/key_pair.hpp>
#include <hpx/components/security/signed_type.hpp>

namespace hpx { namespace components { namespace security { namespace server
{
    class HPX_COMPONENT_EXPORT certificate_authority_base
      : public abstract_fixed_component_base<certificate_authority_base>
    {
    public:
        certificate_authority_base();
        certificate_authority_base(key_pair const &);
        virtual ~certificate_authority_base();

        virtual signed_type<certificate> sign_certificate_signing_request(
            signed_type<certificate_signing_request> const &) const = 0;

        signed_type<certificate> sign_certificate_signing_request_nonvirt(
            signed_type<certificate_signing_request> const & signed_csr)
        {
            return sign_certificate_signing_request(signed_csr);
        }

        signed_type<certificate> get_certificate() const;

        bool is_valid() const;

        HPX_DEFINE_COMPONENT_ACTION(
            certificate_authority_base
          , sign_certificate_signing_request_nonvirt
          , sign_certificate_signing_request_action);

        HPX_DEFINE_COMPONENT_ACTION(
            certificate_authority_base
          , get_certificate);

        HPX_DEFINE_COMPONENT_ACTION(
            certificate_authority_base
          , is_valid);

        virtual naming::gid_type get_base_gid(
            naming::gid_type const& assign_gid = naming::invalid_gid) const = 0;

    protected:
        key_pair key_pair_;

        signed_type<certificate> certificate_;

        bool valid_;
    };
}}}}

HPX_REGISTER_ACTION_DECLARATION(
    hpx::components::security::server::certificate_authority_base
       ::sign_certificate_signing_request_action
  , certificate_authority_base_sign_certificate_signing_request_action);

HPX_REGISTER_ACTION_DECLARATION(
    hpx::components::security::server::certificate_authority_base::get_certificate_action
  , certificate_authority_base_get_certificate_action);

HPX_REGISTER_ACTION_DECLARATION(
    hpx::components::security::server::certificate_authority_base::is_valid_action
  , certificate_authority_base_is_valid_action);

#endif

#endif
