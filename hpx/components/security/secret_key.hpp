//  Copyright (c) 2013 Jeroen Habraken
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_COMPONENTS_SECURITY_SERVER_SECRET_KEY_HPP
#define HPX_COMPONENTS_SECURITY_SERVER_SECRET_KEY_HPP

#include <hpx/config.hpp>

#if defined(HPX_HAVE_SECURITY)

#include <hpx/exception.hpp>

#include <boost/array.hpp>
#include <boost/io/ios_state.hpp>

#include <sodium.h>

#include "public_key.hpp"
#include "signed_type.hpp"

namespace hpx { namespace components { namespace security
{
#if defined(HPX_MSVC)
#  pragma pack(push, 1)
#endif

    class secret_key
    {
    public:
        secret_key(public_key & public_key)
        {
            crypto_sign_keypair(public_key.bytes_.c_array(), bytes_.c_array());
        }

        template <typename T>
        signed_type<T> sign(T const & type, error_code& ec = throws) const
        {
            signed_type<T> signed_type;
            unsigned long long signed_type_length;

            if (crypto_sign(
                    signed_type.begin(),
                    &signed_type_length,
                    type.begin(),
                    type.size(),
                    bytes_.data()) != 0)
            {
                HPX_THROWS_IF(
                    ec
                  , hpx::security_error
                  , "secret_key::sign"
                  , "Failed to sign type"
                )
                return signed_type;
            }

            if (type.size() + crypto_sign_BYTES != signed_type_length)
            {
                HPX_THROWS_IF(
                    ec
                  , hpx::security_error
                  , "secret_key::sign"
                  , "Signature of incorrect length"
                )
                return signed_type;
            }

            if (&ec != &throws)
                ec = make_success_code();

            return signed_type;
        }

        friend std::ostream & operator<<(std::ostream & os,
                                         secret_key const & secret_key)
        {
            boost::io::ios_flags_saver ifs(os);

            os << "<secret_key \"";

            for (std::size_t i = 0; i < crypto_sign_SECRETKEYBYTES; ++i)
            {
                os << std::hex
                   << std::nouppercase
                   << std::setfill('0')
                   << std::setw(2)
                   << static_cast<unsigned int>(secret_key.bytes_[i]);
            }

            return os << "\">";
        }

    private:
        boost::array<
            unsigned char, crypto_sign_SECRETKEYBYTES
        > bytes_;
    };

#if defined(HPX_MSVC)
#  pragma pack(pop)
#endif
}}}

#endif

#endif
