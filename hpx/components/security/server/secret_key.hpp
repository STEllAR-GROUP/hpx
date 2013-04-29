//  Copyright (c) 2013 Jeroen Habraken
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_COMPONENTS_SECURITY_SERVER_SECRET_KEY_HPP
#define HPX_COMPONENTS_SECURITY_SERVER_SECRET_KEY_HPP

#include <boost/array.hpp>
#include <sodium.h>

#include "public_key.hpp"
#include "signed_type.hpp"

namespace hpx { namespace components { namespace security { namespace server
{
    class secret_key
      : boost::array<
            unsigned char, crypto_sign_SECRETKEYBYTES
        >
    {
    public:
        secret_key(public_key & public_key)
        {
            crypto_sign_keypair(public_key.c_array(), c_array());
        }

        template <typename T>
        bool sign(T const & type, signed_type<T> & signed_type) const
        {
            unsigned long long signed_type_length;

            return crypto_sign(
                reinterpret_cast<unsigned char *>(&signed_type),
                &signed_type_length,
                reinterpret_cast<unsigned char const *>(&type),
                sizeof(type),
                data()) == 0;
        }
    };
}}}}

#endif
