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

        issuer.test();
    }

    void subordinate_certificate_authority::test() const
    {
        std::cout << "subordinate_certificate_authority::test\n";
    }
}}}}
