//  Copyright (c) 2013 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_UTIL_SECURITY_SUB_CERTIFICATE_AUTHORITY_MAY_30_2013_0415PM)
#define HPX_UTIL_SECURITY_SUB_CERTIFICATE_AUTHORITY_MAY_30_2013_0415PM

#if defined(HPX_HAVE_SODIUM)

#include <boost/assert.hpp>
#include <hpx/components/security/server/key_pair.hpp>
#include <hpx/components/security/server/subordinate_certificate_authority.hpp>
#include <hpx/hpx_fwd.hpp>

namespace hpx { namespace util { namespace security
{
    class subordinate_certificate_authority
    {
        typedef components::security::server::subordinate_certificate_authority
            certificate_authority_type;

    public:
        subordinate_certificate_authority()
          : subordinate_certificate_authority_(0)
        {
        }

        ~subordinate_certificate_authority();

        void initialize();

        components::security::server::signed_type<
            components::security::server::certificate
        > get_certificate();

        naming::gid_type get_gid() const;

    private:
        components::security::server::key_pair key_pair_;
        certificate_authority_type* subordinate_certificate_authority_;
    };
}}}

#endif
#endif
