//  Copyright (c) 2013 Jeroen Habraken
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_COMPONENTS_SECURITY_SERVER_CERTIFICATE_SIGNING_REQUEST_HPP
#define HPX_COMPONENTS_SECURITY_SERVER_CERTIFICATE_SIGNING_REQUEST_HPP

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/naming/name.hpp>
#include <hpx/runtime/serialization/serialize.hpp>
#include <hpx/runtime/serialization/array.hpp>

#include <hpx/components/security/capability.hpp>
#include <hpx/components/security/public_key.hpp>

namespace hpx { namespace components { namespace security
{
#if defined(_MSC_VER)
#  pragma pack(push, 1)
#endif

    class certificate_signing_request
    {
    public:
        static std::size_t const capability_size = 8;

        certificate_signing_request()
        {
        }

        certificate_signing_request(naming::gid_type const & subject,
                                    public_key const & subject_public_key,
                                    capability const & capability)
          : subject_(subject)
          , subject_public_key_(subject_public_key)
          , capability_(capability)
        {
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
                certificate_signing_request const & certificate_signing_request)
        {
            return os << "<certificate_signing_request "
                      << certificate_signing_request.subject_
                      << " "
                      << certificate_signing_request.subject_public_key_
                      << " "
                      << certificate_signing_request.capability_
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

        BOOST_CONSTEXPR static std::size_t size()
        {
            return sizeof(certificate_signing_request);
        }

    private:
        friend class hpx::serialization::access;

        template <typename Archive>
        void serialize(Archive & ar, const unsigned int)
        {
            ar & hpx::serialization::make_array(begin(), size());
        }

        naming::gid_type subject_;
        public_key subject_public_key_;

        capability capability_;
    };

#if defined(_MSC_VER)
#  pragma pack(pop)
#endif
}}}

namespace hpx { namespace traits
{
    template <>
    struct is_bitwise_serializable<
            hpx::components::security::certificate_signing_request>
       : mpl::true_
    {};
}}

#endif
