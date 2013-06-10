//  Copyright (c) 2013 Jeroen Habraken
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/components/security/server/certificate_authority_base.hpp>

namespace hpx { namespace components { namespace security { namespace server
{
    certificate_authority_base::certificate_authority_base()
      : valid_(false)
    {
    }

    certificate_authority_base::certificate_authority_base(
        key_pair const & key_pair)
      : key_pair_(key_pair)
      , valid_(false)
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

    bool
    certificate_authority_base::is_valid() const
    {
        return valid_;
    }
}}}}
