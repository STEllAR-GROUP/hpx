//  Copyright (c) 2013 Jeroen Habraken
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/components/security/certificate_authority_base.hpp>
#include <hpx/components/security/server/subordinate_certificate_authority.hpp>

#include <iostream>

namespace hpx { namespace components { namespace security { namespace server
{
    subordinate_certificate_authority::subordinate_certificate_authority()
      : certificate_authority_base()
    {
    }

    subordinate_certificate_authority::subordinate_certificate_authority(
        naming::id_type const & issuer_id)
      : certificate_authority_base()
    {
        security::certificate_authority_base issuer(issuer_id);

        signed_type<certificate_signing_request> signed_csr =
            key_pair_.sign(certificate_signing_request(
                get_gid(), key_pair_.get_public_key()));

        certificate_ = issuer.sign_certificate_signing_request(signed_csr);
    }

    subordinate_certificate_authority::subordinate_certificate_authority(
        key_pair const & key_pair
      , naming::id_type const & issuer_id)
      : certificate_authority_base(key_pair)
    {
        security::certificate_authority_base issuer(issuer_id);

        signed_type<certificate_signing_request> signed_csr =
            key_pair_.sign(certificate_signing_request(
                get_gid(), key_pair_.get_public_key()));

        certificate_ = issuer.sign_certificate_signing_request(signed_csr);
    }

    signed_type<certificate>
    subordinate_certificate_authority::sign_certificate_signing_request(
        signed_type<certificate_signing_request> const & signed_csr) const
    {
        signed_type<certificate> signed_certificate;

        certificate_signing_request const & csr = signed_csr.get_type();

        if (csr.get_subject_public_key().verify(signed_csr))
        {
            // TODO, capability checks

            signed_certificate = key_pair_.sign(certificate(get_gid(), csr));
        }

        return signed_certificate;
    }
}}}}
