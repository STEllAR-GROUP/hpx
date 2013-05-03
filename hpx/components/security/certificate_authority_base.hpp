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

        void test() const
        {
            return this->base_type::test(this->get_gid());
        }
    };
}}}

#endif
