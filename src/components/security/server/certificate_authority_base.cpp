//  Copyright (c) 2013 Jeroen Habraken
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/components/security/server/certificate_authority_base.hpp>

#include <iostream>

namespace hpx { namespace components { namespace security { namespace server
{
    certificate_authority_base::certificate_authority_base()
      : secret_key_(public_key_)
    {
    }

    certificate_authority_base::~certificate_authority_base()
    {
    }

    signed_type<certificate>
    certificate_authority_base::get_certificate() const
    {
        return certificate_;
    }
}}}}
