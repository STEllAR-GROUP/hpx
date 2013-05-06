//  Copyright (c) 2013 Jeroen Habraken
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_COMPONENTS_SECURITY_CERTIFICATE_AUTHORITY_BASE_HPP
#define HPX_COMPONENTS_SECURITY_CERTIFICATE_AUTHORITY_BASE_HPP

#include <hpx/hpx_fwd.hpp>
#include <hpx/include/components.hpp>
#include <hpx/components/security/stubs/certificate_authority_base.hpp>

namespace hpx { namespace components { namespace security
{
    class certificate_authority_base
      : public client_base<
            certificate_authority_base
          , stubs::certificate_authority_base
        >
    {
    private:
        typedef client_base<
            certificate_authority_base
          , stubs::certificate_authority_base
        > base_type;

    public:
        certificate_authority_base(naming::id_type const & gid)
          : base_type(gid)
        {
        }

        certificate_authority_base(future_type const & gid)
          : base_type(gid)
        {
        }

        server::signed_type<server::certificate> sign_certificate_signing_request(
            server::signed_type<server::certificate_signing_request> const & signed_csr) const
        {
            return this->base_type::sign_certificate_signing_request(
                this->get_gid(), signed_csr);
        }

        server::signed_type<server::certificate> get_certificate() const
        {
            return this->base_type::get_certificate(this->get_gid());
        }
    };
}}}

#endif
