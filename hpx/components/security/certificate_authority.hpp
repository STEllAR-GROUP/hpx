//  Copyright (c) 2013 Jeroen Habraken
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_COMPONENTS_SECURITY_CERTIFICATE_AUTHORITY_HPP
#define HPX_COMPONENTS_SECURITY_CERTIFICATE_AUTHORITY_HPP

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/components/client_base.hpp>
#include <hpx/components/security/stubs/certificate_authority.hpp>

namespace hpx { namespace components { namespace security
{
    class certificate_authority
      : public client_base<
            certificate_authority
          , stubs::certificate_authority
        >
    {
    private:
        typedef client_base<
            certificate_authority
          , stubs::certificate_authority
        > base_type;

    public:
        certificate_authority(naming::id_type const & gid = naming::invalid_id)
          : base_type(gid)
        {}

        server::signed_type<server::certificate_signing_request>
        get_certificate_signing_request() const
        {
            return this->base_type::get_certificate_signing_request(this->get_gid());
        }
    };
}}}

#endif
