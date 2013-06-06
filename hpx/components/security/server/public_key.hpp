//  Copyright (c) 2013 Jeroen Habraken
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_COMPONENTS_SECURITY_SERVER_PUBLIC_KEY_HPP
#define HPX_COMPONENTS_SECURITY_SERVER_PUBLIC_KEY_HPP

#include <boost/array.hpp>
#include <boost/io/ios_state.hpp>
#include <boost/serialization/serialization.hpp>
#include <sodium.h>

#include "signed_type.hpp"

namespace hpx { namespace components { namespace security { namespace server
{
#if defined(_MSC_VER)
#  pragma pack(push, 1)
#endif

    class public_key
    {
    public:
        template <typename T>
        bool verify(signed_type<T> const & signed_type) const
        {
            unsigned char type[sizeof signed_type];
            unsigned long long type_length;

            return crypto_sign_open(
                type,
                &type_length,
                reinterpret_cast<unsigned char const *>(&signed_type),
                sizeof signed_type,
                bytes_.data()) == 0;
        }

        friend std::ostream & operator<<(std::ostream & os,
                                         public_key const & public_key)
        {
            boost::io::ios_flags_saver ifs(os);

            os << "<public_key \"";

            for (std::size_t i = 0; i < crypto_sign_PUBLICKEYBYTES; ++i)
            {
                os << std::hex
                   << std::nouppercase
                   << std::setfill('0')
                   << std::setw(2)
                   << static_cast<unsigned int>(public_key.bytes_[i]);
            }

            return os << "\">";
        }

    private:
        friend class boost::serialization::access;

        template <typename Archive>
        void serialize(Archive & ar, const unsigned int)
        {
            ar & bytes_;
        }

        friend class secret_key;

        boost::array<
            unsigned char, crypto_sign_PUBLICKEYBYTES
        > bytes_;
    };

#if defined(_MSC_VER)
#  pragma pack(pop)
#endif
}}}}

#endif
