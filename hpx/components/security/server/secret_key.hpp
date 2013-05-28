//  Copyright (c) 2013 Jeroen Habraken
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_COMPONENTS_SECURITY_SERVER_SECRET_KEY_HPP
#define HPX_COMPONENTS_SECURITY_SERVER_SECRET_KEY_HPP

#include <boost/array.hpp>
#include <boost/serialization/serialization.hpp>
#include <hpx/exception.hpp>
#include <sodium.h>

#include "public_key.hpp"
#include "signed_type.hpp"

namespace hpx { namespace components { namespace security { namespace server
{
    class secret_key
    {
    public:
        secret_key(public_key & public_key)
        {
            crypto_sign_keypair(public_key.bytes_.c_array(), bytes_.c_array());
        }

        template <typename T>
        signed_type<T> sign(T const & type) const
        {
            signed_type<T> signed_type;
            unsigned long long signed_type_length;

            if (crypto_sign(
                    reinterpret_cast<unsigned char *>(&signed_type),
                    &signed_type_length,
                    reinterpret_cast<unsigned char const *>(&type),
                    sizeof type,
                    bytes_.data()) != 0)
            {
                HPX_THROW_EXCEPTION(
                    hpx::security_error
                  , "secret_key::sign"
                  , "Failed to sign type"
                )
            }

            if (sizeof type + crypto_sign_BYTES != signed_type_length)
            {
                HPX_THROW_EXCEPTION(
                    hpx::security_error
                  , "secret_key::sign"
                  , "Signature of incorrect length"
                )
            }

            return signed_type;
        }

    private:
        friend class boost::serialization::access;

        template <typename Archive>
        void serialize(Archive & ar, const unsigned int)
        {
            ar & bytes_;
        }

        boost::array<
            unsigned char, crypto_sign_SECRETKEYBYTES
        > bytes_;
    };
}}}}

#endif
