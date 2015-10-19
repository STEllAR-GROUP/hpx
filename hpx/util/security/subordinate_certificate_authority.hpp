//  Copyright (c) 2013 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_UTIL_SECURITY_SUB_CERTIFICATE_AUTHORITY_MAY_30_2013_0415PM)
#define HPX_UTIL_SECURITY_SUB_CERTIFICATE_AUTHORITY_MAY_30_2013_0415PM

#include <hpx/config/defines.hpp>

#if defined(HPX_HAVE_SODIUM)

#include <hpx/config.hpp>
#include <hpx/runtime/naming/name.hpp>
#include <hpx/components/security/key_pair.hpp>
#include <hpx/components/security/server/subordinate_certificate_authority.hpp>

#include <hpx/util/assert.hpp>

#include <boost/cstdint.hpp>

namespace hpx { namespace util { namespace security
{
    class HPX_EXPORT subordinate_certificate_authority
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

        components::security::signed_certificate_signing_request
            get_certificate_signing_request() const;

        components::security::signed_certificate
            sign_certificate_signing_request(
                components::security::signed_certificate_signing_request const &) const;

        void set_certificate(
            components::security::signed_certificate const &);

        components::security::signed_certificate
            get_certificate(error_code& ec = throws) const;

        naming::gid_type get_gid() const;

        bool is_valid() const;

        components::security::key_pair const & get_key_pair() const;

    private:
        components::security::key_pair key_pair_;
        certificate_authority_type* subordinate_certificate_authority_;
    };

    inline boost::uint64_t
    get_subordinate_certificate_authority_msb(boost::uint32_t locality_id)
    {
        return naming::replace_locality_id(
            HPX_SUBORDINATE_CERTIFICATE_AUTHORITY_MSB
          , locality_id);
    }

    inline naming::gid_type
    get_subordinate_certificate_authority_gid(boost::uint32_t locality_id)
    {
        return naming::gid_type(
            get_subordinate_certificate_authority_msb(locality_id)
          , HPX_SUBORDINATE_CERTIFICATE_AUTHORITY_LSB);
    }
}}}

#endif
#endif
