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
    }

    void root_certificate_authority::test() const
    {
        std::cout << "root_certificate_authority::test\n";
    }
}}}}
