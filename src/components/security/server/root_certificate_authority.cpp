//  Copyright (c) 2013 Jeroen Habraken
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/components/security/server/root_certificate_authority.hpp>

#include <iostream>

namespace hpx { namespace components { namespace security { namespace server
{
    root_certificate_authority::root_certificate_authority()
      : certificate_authority_base()
    {
        certificate_ = secret_key_.sign(
            certificate(get_gid(), get_gid(), public_key_));
    }

    signed_type<certificate>
    root_certificate_authority::sign_certificate_signing_request(
        signed_type<certificate_signing_request> const & signed_csr) const
    {
        signed_type<certificate> signed_certificate;

        certificate_signing_request const & csr = signed_csr.get_type();

        if (csr.get_subject_public_key().verify(signed_csr))
        {
            // TODO, capability checks

            signed_certificate = secret_key_.sign(certificate(get_gid(), csr));
        }

        return signed_certificate;
    }
}}}}
