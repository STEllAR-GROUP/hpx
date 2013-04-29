//  Copyright (c) 2013 Jeroen Habraken
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/components/security/server/certificate_authority.hpp>

#include <iostream>

namespace hpx { namespace components { namespace security { namespace server
{
    certificate_authority::certificate_authority()
      : subject_(hpx::find_here())
      , secret_key_(public_key_)
    {
    }

    certificate_authority::certificate_authority(
        naming::id_type const & subject)
      : subject_(subject)
      , secret_key_(public_key_)
    {
    }

    signed_type<certificate_signing_request>
    certificate_authority::get_certificate_signing_request() const
    {
        signed_type<certificate_signing_request> signed_csr;

        if (secret_key_.sign(certificate_signing_request(subject_, public_key_),
                             signed_csr) == false)
        {
            // TODO, throw
        }

        return signed_csr;
    }
}}}}
