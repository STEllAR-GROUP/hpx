//  Copyright (c) 2013 Jeroen Habraken
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_COMPONENTS_SECURITY_STUBS_CERTIFICATE_AUTHORITY_BASE_HPP
#define HPX_COMPONENTS_SECURITY_STUBS_CERTIFICATE_AUTHORITY_BASE_HPP

#include <hpx/hpx_fwd.hpp>
#include <hpx/include/components.hpp>
#include <hpx/components/security/server/certificate_authority_base.hpp>

namespace hpx { namespace components { namespace security { namespace stubs
{
    class certificate_authority_base
      : public stub_base<
            server::certificate_authority_base
        >
    {
    public:
        static void test(hpx::naming::id_type const & gid)
        {
            return hpx::async<
                server::certificate_authority_base::test_action
            >(gid).get();
        }
    };
}}}}

#endif
