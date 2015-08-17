//  Copyright (c) 2013 Jeroen Habraken
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_fwd.hpp>
#include <hpx/components/security/certificate_authority_base.hpp>
#include <hpx/components/security/server/subordinate_certificate_authority.hpp>
#include <hpx/runtime/naming/name.hpp>

namespace hpx { namespace components { namespace security { namespace server
{
    boost::uint64_t get_subordinate_certificate_authority_msb()
    {
        return naming::replace_locality_id(
            HPX_SUBORDINATE_CERTIFICATE_AUTHORITY_MSB
          , get_locality_id());
    }

    subordinate_certificate_authority::subordinate_certificate_authority()
      : certificate_authority_base()
      , fixed_component_base<subordinate_certificate_authority>(
            get_subordinate_certificate_authority_msb()
          , HPX_SUBORDINATE_CERTIFICATE_AUTHORITY_LSB)
    {
    }

    subordinate_certificate_authority::subordinate_certificate_authority(
        key_pair const & key_pair)
      : certificate_authority_base(key_pair)
      , fixed_component_base<subordinate_certificate_authority>(
            get_subordinate_certificate_authority_msb()
          , HPX_SUBORDINATE_CERTIFICATE_AUTHORITY_LSB)
    {
    }

    subordinate_certificate_authority::subordinate_certificate_authority(
        naming::id_type const & issuer_id)
      : certificate_authority_base()
      , fixed_component_base<subordinate_certificate_authority>(
            get_subordinate_certificate_authority_msb()
          , HPX_SUBORDINATE_CERTIFICATE_AUTHORITY_LSB)
    {
        security::certificate_authority_base issuer(issuer_id);

        set_certificate(issuer.sign_certificate_signing_request(
            get_certificate_signing_request()));
    }

    subordinate_certificate_authority::subordinate_certificate_authority(
        key_pair const & key_pair
      , naming::id_type const & issuer_id)
      : certificate_authority_base(key_pair)
      , fixed_component_base<subordinate_certificate_authority>(
            get_subordinate_certificate_authority_msb()
          , HPX_SUBORDINATE_CERTIFICATE_AUTHORITY_LSB)
    {
        security::certificate_authority_base issuer(issuer_id);

        set_certificate(issuer.sign_certificate_signing_request(
            get_certificate_signing_request()));
    }

    signed_type<certificate_signing_request>
    subordinate_certificate_authority::get_certificate_signing_request() const
    {
        // TODO, request capabilities
        capability capability(0x54);

        return key_pair_.sign(certificate_signing_request(
            get_id().get_gid(), key_pair_.get_public_key(), capability));
    }

    signed_type<certificate>
    subordinate_certificate_authority::sign_certificate_signing_request(
        signed_type<certificate_signing_request> const & signed_csr) const
    {
        certificate_signing_request const & csr = signed_csr.get_type();

        if (csr.get_subject_public_key().verify(signed_csr) == false)
        {
            HPX_THROW_EXCEPTION(
                hpx::security_error
              , "subordinate_certificate_authority::sign_certificate_signing_request"
              , boost::str(boost::format(
                    "The certificate signing request signature is invalid: %1%") %
                    signed_csr)
            )
        }

        capability const & issuer_capability =
            certificate_.get_type().get_capability();
        if (issuer_capability.verify_delegation(csr.get_capability()) == false)
        {
            HPX_THROW_EXCEPTION(
                hpx::security_error
              , "subordinate_certificate_authority::sign_certificate_signing_request"
              , boost::str(boost::format(
                    "The certificate signing request its capabilities can't be delegated: %1% %2%") %
                    issuer_capability % csr.get_capability())
            )
        }

        signed_type<certificate> signed_certificate;

        signed_certificate = key_pair_.sign(certificate(
            get_id().get_gid(), csr));

        return signed_certificate;
    }

    void
    subordinate_certificate_authority::set_certificate(
        signed_type<certificate> const & signed_certificate)
    {
        certificate_ = signed_certificate;

        valid_ = true;
    }
}}}}

