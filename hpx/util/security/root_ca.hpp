//  Copyright (c) 2013 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_UTIL_ROOT_CA_MAY_30_2013_0405PM)
#define HPX_UTIL_ROOT_CA_MAY_30_2013_0405PM

#include <hpx/hpx_fwd.hpp>

#if defined(HPX_HAVE_SECURITY)
#include <hpx/components/security/server/key_pair.hpp>
#include <hpx/components/security/server/root_certificate_authority.hpp>

#include <boost/assert.hpp>

namespace hpx { namespace util { namespace security
{
    class root_ca
    {
        typedef components::security::server::root_certificate_authority
            ca_type;

    public:
        root_ca()
          : key_pair_(0)
          , root_ca_(0)
        {}

        ~root_ca();

        void init();

        naming::gid_type const& get_gid() const;

        bool is_valid() const
        {
            return key_pair_ && root_ca_;
        }

        components::security::server::signed_type<
            components::security::server::certificate> get_certificate();

    private:
        components::security::server::key_pair* key_pair_;
        ca_type* root_ca_;
    };
}}}

#endif
#endif


