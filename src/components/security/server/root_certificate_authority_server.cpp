//  Copyright (c) 2013 Jeroen Habraken
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/components/security/server/root_certificate_authority.hpp>
#include <hpx/exception.hpp>

namespace hpx { namespace components { namespace security { namespace server
{
    root_certificate_authority::root_certificate_authority()
      : certificate_authority_base()
      , fixed_component_base<root_certificate_authority>(
            HPX_ROOT_CERTIFICATE_AUTHORITY_MSB, HPX_ROOT_CERTIFICATE_AUTHORITY_LSB)
    {
        capability capability(
            traits::capability<>::root_certificate_authority_capability);

        naming::gid_type gid = get_id().get_gid();
        certificate_ = key_pair_.sign(certificate(
            gid, gid, key_pair_.get_public_key(), capability));

        valid_ = true;
    }

    root_certificate_authority::root_certificate_authority(
        key_pair const & key_pair)
      : certificate_authority_base(key_pair)
      , fixed_component_base<root_certificate_authority>(
            HPX_ROOT_CERTIFICATE_AUTHORITY_MSB, HPX_ROOT_CERTIFICATE_AUTHORITY_LSB)
    {
        capability capability(
            traits::capability<>::root_certificate_authority_capability);

        naming::gid_type gid = get_id().get_gid();
        certificate_ = key_pair_.sign(certificate(
            gid, gid, key_pair_.get_public_key(), capability));

        valid_ = true;
    }

    signed_type<certificate>
    root_certificate_authority::sign_certificate_signing_request(
        signed_type<certificate_signing_request> const & signed_csr) const
    {
        certificate_signing_request const & csr = signed_csr.get_type();
        if (csr.get_subject_public_key().verify(signed_csr) == false)
        {
            HPX_THROW_EXCEPTION(
                hpx::security_error
              , "root_certificate_authority::sign_certificate_signing_request"
              , boost::str(boost::format(
                    "the certificate signing request signature is invalid: %1%") %
                    signed_csr)
            )
        }

        // The capability checks entail two parts:
        // 1) Is the user allowed to obtain the requested capabilities -> TODO
        // 2) Is the issuer allowed to delegated the requested capabilities

        capability const & issuer_capability =
            certificate_.get_type().get_capability();
        if (issuer_capability.verify_delegation(csr.get_capability()) == false)
        {
            HPX_THROW_EXCEPTION(
                hpx::security_error
              , "subordinate_certificate_authority::sign_certificate_signing_request"
              , boost::str(boost::format(
                    "The certificate signing request its capabilities can't be\
                     delegated: %1% %2%") %
                    issuer_capability % csr.get_capability())
            )
        }

        signed_type<certificate> signed_certificate;

        signed_certificate = key_pair_.sign(certificate(
            get_id().get_gid(), csr));

        return signed_certificate;
    }
}}}}
