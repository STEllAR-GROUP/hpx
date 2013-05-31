//  Copyright (c) 2013 Jeroen Habraken
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_COMPONENTS_SECURITY_SERVER_CERTIFICATE_AUTHORITY_BASE_HPP
#define HPX_COMPONENTS_SECURITY_SERVER_CERTIFICATE_AUTHORITY_BASE_HPP

#include <hpx/hpx_fwd.hpp>
#include <hpx/include/actions.hpp>
#include <hpx/include/components.hpp>

#include "certificate.hpp"
#include "certificate_signing_request.hpp"
#include "key_pair.hpp"
#include "signed_type.hpp"

namespace hpx { namespace components { namespace security { namespace server
{
    class HPX_COMPONENT_EXPORT certificate_authority_base
      : public abstract_simple_component_base<certificate_authority_base>
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

        virtual signed_type<certificate> get_certificate() const;

        signed_type<certificate> get_certificate_nonvirt() const
        {
            return get_certificate();
        }

        HPX_DEFINE_COMPONENT_CONST_ACTION(
            certificate_authority_base
          , sign_certificate_signing_request_nonvirt
          , sign_certificate_signing_request_action);

        HPX_DEFINE_COMPONENT_CONST_ACTION(
            certificate_authority_base
          , get_certificate_nonvirt
          , get_certificate_action);

    protected:
        key_pair key_pair_;

        signed_type<certificate> certificate_;
    };
}}}}

HPX_REGISTER_ACTION_DECLARATION(
    hpx::components::security::server::certificate_authority_base::sign_certificate_signing_request_action
  , certificate_authority_base_sign_certificate_signing_request_action);

HPX_REGISTER_ACTION_DECLARATION(
    hpx::components::security::server::certificate_authority_base::get_certificate_action
  , certificate_authority_base_get_certificate_action);

#endif
