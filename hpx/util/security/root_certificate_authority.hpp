//  Copyright (c) 2013 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_UTIL_ROOT_CERTIFICATE_AUTHORITY_MAY_30_2013_0405PM)
#define HPX_UTIL_ROOT_CERTIFICATE_AUTHORITY_MAY_30_2013_0405PM

#include <hpx/config.hpp>

#if defined(HPX_HAVE_SODIUM)
#include <hpx/exception_fwd.hpp>
#include <hpx/components/security/key_pair.hpp>
#include <hpx/components/security/server/root_certificate_authority.hpp>
#include <hpx/runtime/naming/name.hpp>
#include <hpx/util/assert.hpp>

namespace hpx { namespace util { namespace security
{
    class HPX_EXPORT root_certificate_authority
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

        components::security::signed_certificate
            sign_certificate_signing_request(
                components::security::signed_certificate_signing_request const &) const;

        components::security::signed_certificate
            get_certificate(error_code& ec = throws) const;

        static naming::gid_type get_gid();

        bool is_valid() const;

    private:
        components::security::key_pair* key_pair_;
        certificate_authority_type* root_certificate_authority_;
    };
}}}

#endif
#endif
