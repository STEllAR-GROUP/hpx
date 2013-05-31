//  Copyright (c) 2013 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_UTIL_SECURITY_SUB_CA_MAY_30_2013_0415PM)
#define HPX_UTIL_SECURITY_SUB_CA_MAY_30_2013_0415PM

#include <hpx/hpx_fwd.hpp>

#if defined(HPX_HAVE_SECURITY)

#include <hpx/components/security/server/key_pair.hpp>
#include <hpx/components/security/server/subordinate_certificate_authority.hpp>

#include <boost/assert.hpp>

namespace hpx { namespace util { namespace security
{
    class sub_ca
    {
        typedef components::security::server::subordinate_certificate_authority
            ca_type;

    public:
        sub_ca()
          : sub_ca_(0)
        {}

        ~sub_ca();

        void init();

        naming::gid_type get_gid() const;

        bool is_valid() const
        {
            return 0 != sub_ca_ ? true : false;
        }

        components::security::server::signed_type<
            components::security::server::certificate> get_certificate();

    private:
        components::security::server::key_pair key_pair_;
        ca_type* sub_ca_;
    };
}}}

#endif
#endif


