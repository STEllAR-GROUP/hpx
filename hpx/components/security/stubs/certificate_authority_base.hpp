//  Copyright (c) 2013 Jeroen Habraken
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_COMPONENTS_SECURITY_STUBS_CERTIFICATE_AUTHORITY_BASE_HPP
#define HPX_COMPONENTS_SECURITY_STUBS_CERTIFICATE_AUTHORITY_BASE_HPP

#include <hpx/config.hpp>

#if defined(HPX_HAVE_SECURITY)

#include <hpx/components/security/server/certificate_authority_base.hpp>

namespace hpx { namespace components { namespace security { namespace stubs
{
    class certificate_authority_base
      : public stub_base<
            server::certificate_authority_base
        >
    {
    public:
        static signed_type<certificate> sign_certificate_signing_request(
            hpx::naming::id_type const & gid
          , signed_type<certificate_signing_request> const & signed_csr)
        {
            return hpx::async<
                server::certificate_authority_base
                      ::sign_certificate_signing_request_action
            >(gid, signed_csr).get();
        }

        static signed_type<certificate> get_certificate(naming::id_type const & gid)
        {
            return hpx::async<
                server::certificate_authority_base::get_certificate_action
            >(gid).get();
        }

        static bool is_valid(naming::id_type const & gid)
        {
            return hpx::async<
                server::certificate_authority_base::is_valid_action
            >(gid).get();
        }
    };
}}}}

#endif

#endif
