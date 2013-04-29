//  Copyright (c) 2013 Jeroen Habraken
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_COMPONENTS_SECURITY_STUBS_CERTIFICATE_AUTHORITY_HPP
#define HPX_COMPONENTS_SECURITY_STUBS_CERTIFICATE_AUTHORITY_HPP

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/components/stubs/stub_base.hpp>
#include <hpx/components/security/server/certificate_authority.hpp>

namespace hpx { namespace components { namespace security { namespace stubs
{
    class certificate_authority
      : public stub_base<
            server::certificate_authority
        >
    {
    public:
        static server::signed_type<server::certificate_signing_request>
        get_certificate_signing_request(
            hpx::naming::id_type const & gid)
        {
            return hpx::async<
                server::certificate_authority::get_certificate_signing_request_action
            >(gid).get();
        }
    };
}}}}

#endif
