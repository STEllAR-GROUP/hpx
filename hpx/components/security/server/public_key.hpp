//  Copyright (c) 2013 Jeroen Habraken
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_COMPONENTS_SECURITY_SERVER_PUBLIC_KEY_HPP
#define HPX_COMPONENTS_SECURITY_SERVER_PUBLIC_KEY_HPP

#include <boost/array.hpp>
#include <sodium.h>

#include "signed_type.hpp"

namespace hpx { namespace components { namespace security { namespace server
{
    class public_key
    {
    public:
        template <typename T>
        bool verify(signed_type<T> const & signed_type) const
        {
            unsigned char type[sizeof(signed_type)];
            unsigned long long type_length;

            return crypto_sign_open(
                type,
                &type_length,
                reinterpret_cast<unsigned char const *>(&signed_type),
                sizeof(signed_type),
                bytes_.data()) == 0;
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
}}}}

#endif
