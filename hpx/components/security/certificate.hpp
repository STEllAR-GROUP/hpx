//  Copyright (c) 2013 Jeroen Habraken
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_COMPONENTS_SECURITY_SERVER_CERTIFICATE_HPP
#define HPX_COMPONENTS_SECURITY_SERVER_CERTIFICATE_HPP

#include <hpx/config.hpp>

#if defined(HPX_HAVE_SECURITY)

#include <hpx/runtime/naming/name.hpp>
#include <hpx/traits/is_bitwise_serializable.hpp>
#include <hpx/runtime/serialization/serialize.hpp>
#include <hpx/runtime/serialization/array.hpp>

#include "certificate_signing_request.hpp"
#include "public_key.hpp"

#include <boost/io/ios_state.hpp>

namespace hpx { namespace components { namespace security
{
#if defined(HPX_MSVC)
#  pragma pack(push, 1)
#endif

    class certificate
    {
    public:
        certificate()
        {
        }

        certificate(naming::gid_type const & issuer,
                    naming::gid_type const & subject,
                    public_key const & subject_public_key,
                    capability const & capability)
          : issuer_(issuer)
          , subject_(subject)
          , subject_public_key_(subject_public_key)
          , capability_(capability)
        {
        }

        certificate(naming::gid_type const & issuer,
                    certificate_signing_request const & csr)
          : issuer_(issuer)
          , subject_(csr.get_subject())
          , subject_public_key_(csr.get_subject_public_key())
          , capability_(csr.get_capability())
        {
        }

        naming::gid_type const & get_issuer() const
        {
            return issuer_;
        }

        naming::gid_type const & get_subject() const
        {
            return subject_;
        }

        public_key const & get_subject_public_key() const
        {
            return subject_public_key_;
        }

        capability const & get_capability() const
        {
            return capability_;
        }

        friend std::ostream & operator<<(std::ostream & os,
                                         certificate const & certificate)
        {
            return os << "<certificate "
                      << certificate.issuer_
                      << " "
                      << certificate.subject_
                      << " "
                      << certificate.subject_public_key_
                      << " "
                      << certificate.capability_
                      << ">";
        }

        unsigned char const* begin() const
        {
            return reinterpret_cast<unsigned char const*>(this);
        }
        unsigned char const* end() const
        {
            return reinterpret_cast<unsigned char const*>(this) + size();
        }

        HPX_CONSTEXPR static std::size_t size()
        {
            return sizeof(certificate);
        }

    private:
        friend class hpx::serialization::access;

        template <typename Archive>
        void serialize(Archive & ar, const unsigned int)
        {
            ar & hpx::serialization::make_array(begin(), size());
        }

        naming::gid_type issuer_;

        naming::gid_type subject_;
        public_key subject_public_key_;

        capability capability_;
    };

#if defined(HPX_MSVC)
#  pragma pack(pop)
#endif
}}}

HPX_IS_BITWISE_SERIALIZABLE(hpx::components::security::certificate)

#endif

#endif
