//  Copyright (c) 2013 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_UTIL_ROOT_CERTIFICATE_AUTHORITY_MAY_30_2013_0405PM)
#define HPX_UTIL_ROOT_CERTIFICATE_AUTHORITY_MAY_30_2013_0405PM

#if defined(HPX_HAVE_SODIUM)

#include <boost/assert.hpp>
#include <hpx/components/security/server/key_pair.hpp>
#include <hpx/components/security/server/root_certificate_authority.hpp>
#include <hpx/hpx_fwd.hpp>

namespace hpx { namespace util { namespace security
{
    class root_certificate_authority
    {
        typedef components::security::server::root_certificate_authority
            certificate_authority_type;

    public:
        root_certificate_authority()
          : key_pair_(0)
          , root_certificate_authority_(0)
        {
        }

        ~root_certificate_authority();

        void initialize();

        components::security::server::signed_type<
            components::security::server::certificate
        > get_certificate();

        static naming::gid_type get_gid();

    private:
        components::security::server::key_pair* key_pair_;
        certificate_authority_type* root_certificate_authority_;
    };
}}}

#endif
#endif
